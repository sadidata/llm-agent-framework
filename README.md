<div align="center">

# 🤖 LLM Agent Framework

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain-ai.github.io/langgraph)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Autonomous LLM agent framework with tool use, persistent memory, ReAct planning and multi-agent orchestration.**

*Built by [Abdulaziz Sadi-Cherif](https://github.com/sadidata) · SADIDATA · Paris 🇫🇷*

</div>

---

## 🚀 Overview

`llm-agent-framework` is a production-grade framework for building autonomous AI agents. Agents can use tools (web search, code execution, APIs, databases), maintain persistent memory across sessions, plan multi-step tasks using ReAct/CoT, and collaborate in multi-agent pipelines via LangGraph.

**Agent Loop:** Perceive → Think (LLM) → Plan → Act (Tool) → Observe → Repeat

---

## ✨ Features

- 🧠 **ReAct & CoT Planning** — Chain-of-thought reasoning before every action
- - 🛠️ **Pluggable Tools** — Web search, Python REPL, SQL, APIs, file system
  - - 💾 **Persistent Memory** — Short-term (conversation) + long-term (vector store)
    - - 🔗 **Multi-Agent Graphs** — Orchestrator/worker patterns via LangGraph
      - - 📊 **Structured Output** — Pydantic-validated tool calls and responses
        - - 🔍 **Full Observability** — Token usage, latency, tool traces via Langfuse
          - - 🔒 **Safety Layer** — Tool sandboxing and output validation
           
            - ---

            ## 📁 Project Structure

            ```
            llm-agent-framework/
            ├── src/
            │   ├── agents/
            │   │   ├── base.py             # BaseAgent abstract class
            │   │   ├── react_agent.py      # ReAct planning agent
            │   │   ├── planner_agent.py    # Task decomposition agent
            │   │   └── orchestrator.py     # Multi-agent orchestrator
            │   ├── tools/
            │   │   ├── registry.py         # Tool registry & discovery
            │   │   ├── web_search.py       # Tavily/SerpAPI web search
            │   │   ├── python_repl.py      # Sandboxed Python executor
            │   │   ├── sql_tool.py         # Database query tool
            │   │   └── file_tool.py        # File read/write tool
            │   ├── memory/
            │   │   ├── short_term.py       # Conversation buffer memory
            │   │   └── long_term.py        # Vector store episodic memory
            │   ├── graphs/
            │   │   ├── react_graph.py      # LangGraph ReAct workflow
            │   │   └── multi_agent.py      # Multi-agent collaboration graph
            │   └── utils/
            │       ├── config.py
            │       └── tracing.py
            ├── examples/
            │   ├── data_analyst_agent.py   # Autonomous data analysis
            │   ├── research_agent.py       # Web research pipeline
            │   └── coding_agent.py         # Code generation & execution
            ├── tests/
            ├── requirements.txt
            └── README.md
            ```

            ---

            ## 💡 Quick Start

            ```python
            from src.agents.react_agent import ReActAgent
            from src.tools.registry import ToolRegistry
            from src.tools.web_search import WebSearchTool
            from src.tools.python_repl import PythonReplTool
            from src.memory.long_term import LongTermMemory

            # Register tools
            registry = ToolRegistry()
            registry.register(WebSearchTool(provider="tavily"))
            registry.register(PythonReplTool(sandbox=True))

            # Create agent with persistent memory
            memory = LongTermMemory(collection="agent_sessions")
            agent = ReActAgent(
                model="gpt-4o",
                tools=registry.get_all(),
                memory=memory,
                max_iterations=10,
            )

            # Run autonomous task
            result = agent.run(
                task="""
                Analyze the latest trends in electric vehicle adoption in France.
                Find data, create a summary table, and generate a Python chart.
                """,
                session_id="session_001",
            )

            print(result.output)
            print(f"Tools used: {result.tool_calls}")
            print(f"Iterations: {result.iterations}")
            ```

            ---

            ## 🔗 Multi-Agent Example

            ```python
            from src.agents.orchestrator import MultiAgentOrchestrator
            from src.agents.react_agent import ReActAgent

            # Define specialist agents
            researcher = ReActAgent(model="gpt-4o", tools=[web_search], name="researcher")
            analyst = ReActAgent(model="gpt-4o", tools=[python_repl, sql], name="analyst")
            writer = ReActAgent(model="gpt-4o", tools=[file_tool], name="writer")

            # Orchestrate
            orchestrator = MultiAgentOrchestrator(
                agents=[researcher, analyst, writer],
                routing_model="gpt-4o-mini",
            )

            result = orchestrator.run(
                "Write a comprehensive market analysis report on AI startups in 2025"
            )
            ```

            ---

            ## 🛠️ Built-in Tools

            | Tool | Description | Provider |
            |------|-------------|----------|
            | `WebSearchTool` | Real-time web search | Tavily / SerpAPI |
            | `PythonReplTool` | Sandboxed Python execution | RestrictedPython |
            | `SQLTool` | Natural language to SQL | SQLAlchemy |
            | `FileTool` | Read/write local files | stdlib |
            | `APITool` | Generic HTTP API calls | httpx |
            | `RAGTool` | Search internal documents | ChromaDB |

            ---

            ## 📊 Agent Trace Example

            ```
            [TASK] Analyze EV adoption trends in France

            [ITER 1] 🤔 Thinking: I need recent data on EV adoption. I'll search the web.
            [ITER 1] 🔧 Tool: web_search("EV adoption France 2024 statistics")
            [ITER 1] 👁️ Observation: Found 8 relevant results...

            [ITER 2] 🤔 Thinking: I have data. I'll write Python to parse and visualize it.
            [ITER 2] 🔧 Tool: python_repl(code="""...""")
            [ITER 2] 👁️ Observation: Chart saved to output/ev_france.png

            [ITER 3] 🤔 Thinking: Analysis complete. I'll compile the final report.
            [DONE] ✅ Task completed in 3 iterations (12.4s, 2,847 tokens)
            ```

            ---

            ## 📦 Requirements

            ```
            langgraph>=0.1.0
            langchain>=0.2.0
            langchain-openai>=0.1.0
            langchain-anthropic>=0.1.0
            tavily-python>=0.3.0
            restrictedpython>=7.0.0
            sqlalchemy>=2.0.0
            chromadb>=0.5.0
            pydantic>=2.7.0
            langfuse>=2.0.0
            ```

            ---

            <div align="center">

            Made with ❤️ by **Abdulaziz Sadi-Cherif** | [GitHub](https://github.com/sadidata) | [Email](mailto:sadidataconseil@gmail.com)

            </div>
