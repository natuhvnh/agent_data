import os
import json
import time
import functools
import operator
from typing import Annotated, Literal, Optional, List, Dict, Any, Type
from langchain.agents import create_agent
from langgraph.graph import MessagesState, END, START, StateGraph
from langgraph.types import Command
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_tavily import TavilySearch
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from llmclean import strip_fences
import matplotlib.pyplot as plt
from azure.cosmos import CosmosClient
from cosmos_agent import CosmosRouteAgent, CosmosOrderAgent
from prompts import plan_prompt, executor_prompt, agent_system_prompt, sum_token_usage
from helper import python_repl_tool

#
from dotenv import load_dotenv

load_dotenv()


# Custom State class with specific keys
class State(MessagesState):
    user_query: Optional[str]  # The user's original query
    enabled_agents: Optional[
        List[str]
    ]  # Makes our multi-agent system modular on which agents to include
    plan: Optional[
        List[Dict[int, Dict[str, Any]]]
    ]  # Listing the steps in the plan needed to achieve the goal.
    current_step: int  # Marking the current step in the plan.
    agent_query: Optional[
        str
    ]  # Inbox note: `agent_query` tells the next agent exactly what to do at the current step.
    last_reason: Optional[
        str
    ]  # Explains the executor’s decision to help maintain continuity and provide traceability.
    replan_flag: Optional[
        bool
    ]  # Set by the executor to indicate that the planner should revise the plan.
    replan_attempts: Optional[
        Dict[int, Dict[int, int]]
    ]  # Replan attempts tracked per step number.
    chart_b64: Optional[str]  # data URI of the cosmos agent's chart, for the front-end
    node_timings: Annotated[
        List[Dict[str, Any]], operator.add
    ]  # per-node wall-time records
    token_usage: Annotated[List[Dict[str, Any]], operator.add]  # per-node token counts


#
def planner_node(state: State) -> Command[Literal["executor"]]:
    """
    Runs the planning LLM and stores the resulting plan in state.
    """
    # 1. Invoke LLM with the planner prompt
    llm_reply = reasoning_llm.invoke([plan_prompt(state)])
    output = llm_reply.content
    output = strip_fences(output)  # remove markdown json in the output
    # 2. Validate JSON
    try:
        content_str = output if isinstance(output, str) else str(output)
        parsed_plan = json.loads(content_str)
    except json.JSONDecodeError:
        raise ValueError(f"Planner returned invalid JSON:\n{output}")

    # 3. Store as current plan only
    replan = state.get("replan_flag", False)
    updated_plan: Dict[str, Any] = parsed_plan
    return Command(
        update={
            "plan": updated_plan,
            "messages": [
                HumanMessage(
                    content=llm_reply.content,
                    name="replan" if replan else "initial_plan",
                )
            ],
            "user_query": state.get("user_query", state["messages"][0].content),
            "current_step": 1 if not replan else state["current_step"],
            # Preserve replan flag so executor runs planned agent once before reconsidering
            "replan_flag": state.get("replan_flag", False),
            "last_reason": "",
            "enabled_agents": state.get("enabled_agents"),
            "token_usage": [{"node": "planner", **sum_token_usage([llm_reply])}],
        },
        goto="executor",
    )


#
def executor_node(
    state: State, MAX_REPLANS=3
) -> Command[Literal["planner", "cosmos_order", "cosmos_route", "synthesizer"]]:

    plan: Dict[str, Any] = state.get("plan", {})
    step: int = state.get("current_step", 1)

    # All plan steps executed — hand off to synthesizer
    if step > len(plan):
        return Command(update={"current_step": step}, goto="synthesizer")

    # 0) If we *just* replanned, run the planned agent once before reconsidering.
    if state.get("replan_flag"):
        planned_agent = plan.get(str(step), {}).get("agent")
        return Command(
            update={
                "replan_flag": False,
                "current_step": step + 1,  # advance because executed the planned agent
            },
            goto=planned_agent,
        )

    # 1) Build prompt & call LLM
    llm_reply = reasoning_llm.invoke([executor_prompt(state)])
    output = llm_reply.content
    output = strip_fences(output)  # remove markdown json in the output
    try:
        content_str = output if isinstance(output, str) else str(output)
        parsed = json.loads(content_str)
        replan: bool = parsed["replan"]
        goto: str = parsed["goto"]
        reason: str = parsed["reason"]
        query: str = parsed["query"]
    except Exception as exc:
        raise ValueError(f"Invalid executor JSON:\n{llm_reply.content}") from exc

    # Update the state
    updates: Dict[str, Any] = {
        "messages": [HumanMessage(content=llm_reply.content, name="executor")],
        "last_reason": reason,
        "agent_query": query,
        "token_usage": [{"node": "executor", **sum_token_usage([llm_reply])}],
    }

    # Replan accounting
    replans: Dict[int, int] = state.get("replan_attempts", {}) or {}
    step_replans = replans.get(step, 0)

    # 2) Replan decision
    if replan:
        if step_replans < MAX_REPLANS:
            replans[step] = step_replans + 1
            updates.update(
                {
                    "replan_attempts": replans,
                    "replan_flag": True,  # ensure next turn executes the planned agent once
                    "current_step": step,  # stay on same step for the new plan
                }
            )
            return Command(update=updates, goto="planner")
        else:
            # Cap hit: skip this step; let next step (or synthesizer) handle termination
            next_agent = plan.get(str(step + 1), {}).get("agent", "synthesizer")
            updates["current_step"] = step + 1
            return Command(update=updates, goto=next_agent)

    # 3) Happy path: run chosen agent; advance only if following the plan
    planned_agent = plan.get(str(step), {}).get("agent")
    updates["current_step"] = step + 1 if goto == planned_agent else step
    updates["replan_flag"] = False
    return Command(update=updates, goto=goto)


#
def synthesizer_node(state: State) -> Command[Literal[END]]:
    """
    Creates a concise, human‑readable summary of the entire interaction, **purely in prose**.
    It ignores structured tables or chart IDs and instead rewrites the relevant agent messages (research results, chart commentary, etc.) into a short final answer.
    """
    # Gather informative messages for final synthesis
    relevant_msgs = [
        m.content
        for m in state.get("messages", [])
        if getattr(m, "name", None) in ("cosmos_order", "cosmos_route")
    ]
    user_question = state.get(
        "user_query",
        state.get("messages", [{}])[0].content if state.get("messages") else "",
    )
    synthesis_instructions = """
        You are the Synthesizer. Use the context below to directly answer the user's question.
        Perform any lightweight calculations, comparisons, or inferences required.
        Do not invent facts not supported by the context.
        If data is missing, say what's missing and, if helpful, offer a clearly labeled best-effort estimate with assumptions.
        Produce a concise response that fully answers the question, with the following guidance:
        - Start with the direct answer (one short paragraph or a tight bullet list).\n
        - Include key figures from any 'Results:' tables (e.g., totals, top items).\n
        - If any message contains citations, include them as a brief 'Citations: [...]' line.\n
        - Keep the output crisp; avoid meta commentary or tool instructions.
        - Return plain text only, do not use markdown.
        """
    summary_prompt = [
        HumanMessage(
            content=(
                f"User question: {user_question}\n\n"
                f"{synthesis_instructions}\n\n"
                f"Context:\n\n" + "\n\n---\n\n".join(relevant_msgs)
            )
        )
    ]

    llm_reply = reasoning_llm.invoke(summary_prompt)

    answer = llm_reply.content.strip()
    # print(f"Synthesizer answer: {answer}")

    return Command(
        update={
            "final_answer": answer,
            "messages": [HumanMessage(content=answer, name="synthesizer")],
            "token_usage": [{"node": "synthesizer", **sum_token_usage([llm_reply])}],
        },
        goto=END,  # hand off to the END node
    )


#
def token_summary(final_state):
    records = final_state.get("token_usage") or []
    print("--- Token Usage by Node ---")
    for rec in records:
        print(
            f"  {rec['node']}: in={rec.get('input_tokens', 0):,}  out={rec.get('output_tokens', 0):,}"
        )
    total_in = sum(r.get("input_tokens", 0) for r in records)
    total_out = sum(r.get("output_tokens", 0) for r in records)
    print(f"--- Total Token Usage ---")
    print(f"Input Tokens:  {total_in:,}")
    print(f"Output Tokens: {total_out:,}")
    print(f"Total Tokens:  {total_in + total_out:,}")


#
def timed_node(name: str, fn):
    """Wraps a node function to record its wall-clock execution time into state."""

    @functools.wraps(fn)
    def wrapper(state):
        start = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - start
        print(f"[timing] {name}: {elapsed:.2f}s")
        if isinstance(result, Command):
            if result.update is None:
                result.update = {}
            result.update["node_timings"] = [{"node": name, "seconds": elapsed}]
        return result

    return wrapper


#
def timing_summary(final_state, total_seconds=None):
    timings = final_state.get("node_timings") or []
    print("--- Node Execution Times ---")
    for rec in timings:
        print(f"{rec['node']}: {rec['seconds']:.2f}s")
    print(f"Sum of node times: {sum(r['seconds'] for r in timings):.2f}s")
    if total_seconds is not None:
        print(f"Total run time:    {total_seconds:.2f}s")


if __name__ == "__main__":
    KEY = os.getenv("cosmos_key")
    ENDPOINT = os.getenv("cosmos_url")
    openai_key = os.getenv("gemini_key")
    azure_openai_key = os.getenv("azure_openai_key")
    tavily_key = os.getenv("tavily_key")
    #
    reasoning_llm = ChatOpenAI(
        model="DeepSeek-V4-Pro",  # DeepSeek-V4-Pro, DeepSeek-V4-Flash
        base_url="https://3t-ai-resource.services.ai.azure.com/openai/v1",
        api_key=azure_openai_key,
        max_tokens=2048,
        temperature=0.1,
    )
    #
    cosmos_order_agent = CosmosOrderAgent(
        endpoint=ENDPOINT,
        key=KEY,
        database_name="hgs-input",
        container_name="orders",
        llm=reasoning_llm,
    )

    cosmos_route_agent = CosmosRouteAgent(
        endpoint=ENDPOINT,
        key=KEY,
        database_name="hgs-output",
        container_name="route",
        llm=reasoning_llm,
    )
    #
    workflow = StateGraph(State)
    workflow.add_node("planner", timed_node("planner", planner_node))
    workflow.add_node("executor", timed_node("executor", executor_node))
    workflow.add_node(
        "cosmos_order", timed_node("cosmos_order", cosmos_order_agent.node)
    )
    workflow.add_node(
        "cosmos_route", timed_node("cosmos_route", cosmos_route_agent.node)
    )
    workflow.add_node("synthesizer", timed_node("synthesizer", synthesizer_node))
    workflow.add_edge(START, "planner")
    graph = workflow.compile()
    #
    query = """
    Calculate the average number of routes planned each calendar day for April 2026, do not return only the overall average and follow output example below. Visualize the results by a line chart.
    <Example output from query>
    April 8: 8 routes
    April 10: 15 routes
    <Example output from query>
    """
    # query = """
    # Give me the number of routes for the request id: cd847bd2-84d1-4132-83b9-f36b3e490a66
    # """
    state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "enabled_agents": ["cosmos_route", "cosmos_order", "synthesizer"],
    }
    import base64

    _run_start = time.perf_counter()
    final_state = graph.invoke(state)
    _total = time.perf_counter() - _run_start
    final_answer = final_state["messages"][-1].content
    print(final_answer)
    print("--------------------------------")
    token_summary(final_state)
    timing_summary(final_state, _total)
    # Display the chart inline if the cosmos agent produced one
    chart_b64 = final_state.get("chart_b64")
    print("=" * 150)
    print(final_state)
    if chart_b64:
        chart_path = os.path.join(os.path.dirname(__file__), "chart_output.png")
        with open(chart_path, "wb") as f:
            f.write(base64.b64decode(chart_b64.split(",", 1)[1]))
        print(f"\n--- Chart saved to {chart_path} ---")
        img = plt.imread(chart_path)
        plt.imshow(img)
        plt.show()
    else:
        print("(No chart produced)")
