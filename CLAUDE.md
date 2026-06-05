# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-agent data analysis system built on **LangGraph**. It accepts a natural-language query, produces a step-by-step execution plan via an LLM, then routes each step to a specialized sub-agent (Cosmos DB query for orders or routes, synthesis). The primary use case is querying Azure Cosmos DB logistics data (optimizer inputs and outputs) and returning prose answers, optionally with matplotlib chart visualizations.

## Running the Project

There is no build system, Makefile, or test suite. All dependencies must be installed manually (see imports in source files for the full list).

**Run the main pipeline:**
```bash
python3 data_agent.py
```
This executes the hard-coded query at the bottom of `data_agent.py` (the `if __name__ == "__main__"` block), prints the final answer + token usage, and saves `chart_output.png` if a chart was produced.

**Key libraries to install:**
```
langgraph langchain langchain-core langchain-openai langchain-community
langchain-tavily azure-cosmos python-dotenv pydantic matplotlib numpy llmclean
```

> `helper.py` also imports `snowflake-snowpark-python`, `trulens`, and `langchain-experimental` for TruLens observability and Snowflake Cortex research. These are only needed for notebook-based exploration, not for the main pipeline.

**Interactive exploration:** `search_agent.ipynb` and `cosmos_test.ipynb`.

## Architecture

### LangGraph StateGraph (`data_agent.py`)

State type (`State`, line 32): extends `MessagesState` with the following custom fields:

| Field | Type | Purpose |
|---|---|---|
| `user_query` | `Optional[str]` | The user's original query |
| `enabled_agents` | `Optional[List[str]]` | Controls which agents are available (modular) |
| `plan` | `Optional[Dict[str, Dict[str, Any]]]` | Step-by-step plan (step number → step definition) |
| `current_step` | `int` | Index of the current plan step being executed |
| `agent_query` | `Optional[str]` | Exact instruction passed to the next agent |
| `last_reason` | `Optional[str]` | Executor's reasoning for traceability |
| `replan_flag` | `Optional[bool]` | Signals the planner to revise the plan |
| `replan_attempts` | `Optional[Dict[int, int]]` | Replan attempt count tracked per step |
| `chart_b64` | `Optional[str]` | Base64-encoded PNG data URI for front-end display |

Nodes wired in the graph:
| Node | Source | Role |
|---|---|---|
| `planner` | `data_agent.py` | LLM produces a JSON step-by-step plan |
| `executor` | `data_agent.py` | Reads next plan step, routes via `Command(goto=...)` |
| `cosmos_order` | `cosmos_agent.py` → `CosmosOrderAgent` | ReAct agent querying Cosmos DB orders (optimizer input) |
| `cosmos_route` | `cosmos_agent.py` → `CosmosRouteAgent` | ReAct agent querying Cosmos DB routes (optimizer output) |
| `synthesizer` | `data_agent.py` | Produces final prose answer from agent messages |

Only the `START → planner` edge is static; all other routing is dynamic via `Command(goto=...)` returned by `executor_node`.

### Key Files

- **`data_agent.py`** — Graph definition, all node implementations, LLM config (Azure OpenAI `DeepSeek-V4-Flash`), and the runnable `__main__` block with the hard-coded query.
- **`prompts.py`** — All prompt builders (`plan_prompt`, `executor_prompt`, `agent_system_prompt`) and the **agent registry** (`get_agent_descriptions()`). Change agent behavior, capabilities, and routing logic here.
- **`cosmos_agent.py`** — `CosmosRouteAgent` and `CosmosOrderAgent`, each wrapping a Cosmos DB ReAct agent. Both detect and base64-encode newly generated PNGs after invocation.
- **`helper.py`** — `python_repl_tool`: writes LLM-generated chart code to `temp_agent_script.py` and executes it via `subprocess.run`. Also contains TruLens observability setup and Snowflake utilities (notebook use only).

### Non-Obvious Details

- **`temp_agent_script.py`** is machine-generated on every run — do not treat it as source.
- `MAX_REPLANS` is `3` in `data_agent.py` (as a default parameter in `executor_node`) but `2` in `prompts.py` — these are inconsistent. The effective limit during execution is `3`; the prompt text says `2`.
- `helper.py` imports heavy optional deps (`snowflake`, `trulens`) at module load time. If those packages aren't installed, importing `helper.py` will fail even for the main pipeline.
- Chart code is LLM-generated, written to `temp_agent_script.py`, executed as a subprocess, then the PNG is base64-encoded and the disk file is deleted by `cosmos_agent.py`.
- `CosmosRouteAgent` queries database `hgs-output`, container `route`. `CosmosOrderAgent` queries database `hgs-input`, container `orders` (schema says `route_request` internally — check for drift).
- `synthesizer_node` only incorporates messages from `cosmos_order` and `cosmos_route` named senders; messages from other agents are silently excluded.

## Configuration

Credentials are loaded from `.env` via `load_dotenv()`. Required variables:

| Variable | Used for |
|---|---|
| `azure_openai_key` | Active LLM (Azure AI / DeepSeek-V4-Flash) |
| `cosmos_url`, `cosmos_key` | Azure Cosmos DB |
| `tavily_key` | Web search via Tavily (unused in current graph; available for future use) |
| `gemini_key` | Alternate Gemini endpoint (currently commented out) |

`.env` is gitignored. `chart_output.png`, `temp_agent_script.py`, `__pycache__/`, and `.ipynb_checkpoints/` are also gitignored.

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
