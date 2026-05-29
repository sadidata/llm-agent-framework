"""ReAct (Reasoning + Acting) agent with LangGraph."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from src.agents.base import BaseAgent, AgentResult
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.utils.tracing import AgentTracer


REACT_SYSTEM_PROMPT = """\
You are an autonomous AI agent. You solve tasks step by step using available tools.

For EVERY step, follow this exact format:

Thought: <your reasoning about what to do next>
Action: <tool name>
Action Input: <tool input as JSON>

After receiving an Observation, continue with another Thought/Action pair,
or if the task is complete:

Thought: I have enough information to provide the final answer.
Final Answer: <your complete answer>

Available tools:
{tool_descriptions}
"""


@dataclass
class AgentState:
      """Mutable state passed through the LangGraph workflow."""
      messages: List[BaseMessage] = field(default_factory=list)
      tool_calls: List[dict] = field(default_factory=list)
      iterations: int = 0
      final_answer: Optional[str] = None
      error: Optional[str] = None


class ReActAgent(BaseAgent):
      """ReAct agent with LangGraph-powered workflow, tools and memory."""

    def __init__(
              self,
              model: str = "gpt-4o",
              tools: Optional[List[BaseTool]] = None,
              memory: Optional[LongTermMemory] = None,
              max_iterations: int = 10,
              temperature: float = 0.0,
              name: str = "react_agent",
    ):
              self.name = name
              self.model_name = model
              self.tools = tools or []
              self.long_term_memory = memory
              self.short_term_memory = ShortTermMemory(window=20)
              self.max_iterations = max_iterations
              self._llm = self._build_llm(model, temperature)
              self._tool_map = {t.name: t for t in self.tools}
              self._graph = self._build_graph()
              self._tracer = AgentTracer(agent_name=name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str, session_id: Optional[str] = None) -> AgentResult:
              """Execute a task autonomously.

                      Args:
                                  task: Natural language task description.
                                              session_id: Optional session ID for persistent memory.

                                                      Returns:
                                                                  AgentResult with output, tool calls and metadata.
                                                                          """
              start_time = time.time()
              trace_id = self._tracer.start_trace(task=task, session_id=session_id)

        # Load relevant long-term memory
              context = ""
              if self.long_term_memory and session_id:
                            memories = self.long_term_memory.retrieve(task, top_k=3)
                            if memories:
                                              context = "\nRelevant past context:\n" + "\n".join(
                                                                    f"- {m}" for m in memories
                                              )

                        tool_desc = self._format_tool_descriptions()
        system_msg = SystemMessage(
                      content=REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)
        )
        human_msg = HumanMessage(content=f"Task: {task}{context}")

        state = AgentState(messages=[system_msg, human_msg])
        final_state = self._graph.invoke(state)

        elapsed = time.time() - start_time
        result = AgentResult(
                      output=final_state.get("final_answer", "No answer produced."),
                      tool_calls=final_state.get("tool_calls", []),
                      iterations=final_state.get("iterations", 0),
                      elapsed_seconds=elapsed,
                      session_id=session_id,
        )

        # Persist to long-term memory
        if self.long_term_memory and session_id:
                      self.long_term_memory.store(
                                        content=f"Task: {task}\nResult: {result.output[:500]}",
                                        metadata={"session_id": session_id, "agent": self.name},
                      )

        self._tracer.end_trace(trace_id, result=result)
        return result

    # ------------------------------------------------------------------
    # LangGraph workflow
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
              graph = StateGraph(dict)

        graph.add_node("think", self._think_node)
        graph.add_node("act", self._act_node)

        graph.set_entry_point("think")
        graph.add_conditional_edges(
                      "think",
                      self._should_act_or_finish,
                      {"act": "act", "finish": END},
        )
        graph.add_edge("act", "think")

        return graph.compile()

    def _think_node(self, state: dict) -> dict:
              """LLM reasoning step."""
              state["iterations"] = state.get("iterations", 0) + 1
              if state["iterations"] > self.max_iterations:
                            state["final_answer"] = "Max iterations reached without completing the task."
                            return state

              response = self._llm.invoke(state["messages"])
              state["messages"].append(response)

        content = response.content
        if "Final Answer:" in content:
                      state["final_answer"] = content.split("Final Answer:")[-1].strip()

        return state

    def _act_node(self, state: dict) -> dict:
              """Tool execution step."""
              last_message = state["messages"][-1].content
              tool_name, tool_input = self._parse_action(last_message)

        if not tool_name:
                      return state

        tool = self._tool_map.get(tool_name)
        if not tool:
                      observation = f"Error: Tool '{tool_name}' not found."
else:
              try:
                                observation = tool.run(tool_input)
except Exception as exc:
                observation = f"Tool error: {exc}"

        state["tool_calls"].append({
                      "tool": tool_name,
                      "input": tool_input,
                      "output": str(observation)[:500],
                      "iteration": state["iterations"],
        })

        obs_msg = HumanMessage(content=f"Observation: {observation}")
        state["messages"].append(obs_msg)
        return state

    def _should_act_or_finish(self, state: dict) -> str:
              if state.get("final_answer"):
                            return "finish"
                        last = state["messages"][-1].content
        if "Action:" in last and "Action Input:" in last:
                      return "act"
                  return "finish"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_llm(self, model: str, temperature: float):
              if model.startswith("gpt"):
                            return ChatOpenAI(model=model, temperature=temperature)
elif model.startswith("claude"):
            return ChatAnthropic(model=model, temperature=temperature)
        raise ValueError(f"Unknown model: {model}")

    def _format_tool_descriptions(self) -> str:
              if not self.tools:
                            return "No tools available."
                        lines = []
        for t in self.tools:
                      lines.append(f"- {t.name}: {t.description}")
                  return "\n".join(lines)

    @staticmethod
    def _parse_action(text: str):
              """Extract tool name and input from ReAct output."""
        try:
                      action_line = [l for l in text.split("\n") if l.startswith("Action:")]
                      input_line = [l for l in text.split("\n") if l.startswith("Action Input:")]
                      if not action_line or not input_line:
                                        return None, None
                                    tool_name = action_line[0].replace("Action:", "").strip()
            tool_input_str = input_line[0].replace("Action Input:", "").strip()
            try:
                              tool_input = json.loads(tool_input_str)
except json.JSONDecodeError:
                tool_input = tool_input_str
            return tool_name, tool_input
except Exception:
            return None, None
