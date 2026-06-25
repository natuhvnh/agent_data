import os
import json
import time
import functools
import asyncio
import inspect
import base64
from typing import Literal, List, Dict, Any
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
)
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_azure_cosmosdb import AzureCosmosDBNoSqlVectorSearch
from langgraph_checkpoint_cosmosdb import CosmosDBSaver
from langgraph.types import Command
from langgraph.graph import MessagesState, END, START, StateGraph
from llmclean import strip_fences
import matplotlib.pyplot as plt
from azure.cosmos import PartitionKey
# from azure.cosmos.aio import CosmosClient
from azure.cosmos import CosmosClient
from azure.core.credentials import AzureKeyCredential
from cosmos_agent import CosmosRouteAgent, CosmosOrderAgent

# from cosmos_agent_mcp import CosmosRouteAgent, CosmosOrderAgent
from prompts import plan_prompt, executor_prompt, sum_token_usage, MAX_REPLANS
from state_message import State

#
from dotenv import load_dotenv

load_dotenv()


async def memory_prep_node(state: State, config: RunnableConfig) -> dict:
    # Notice we removed `*, store: BaseStore` from the signature!

    cfg = config.get("configurable", {})
    user_id = cfg.get("user_id", "anonymous")
    user_query = state.get("user_query") or (
        state["messages"][-1].content if state.get("messages") else ""
    )

    # 1. Reset the LangGraph message list safely
    new_user_msg = state["messages"][-1]
    remove_all = [
        RemoveMessage(id=m.id)
        for m in state.get("messages", [])
        if getattr(m, "id", None)
    ]

    # 2. Long-term recall — using LangChain VectorStore methods
    long_term_memories: List[str] = []
    if store and user_query:
        # --- FIXED: Use asimilarity_search instead of asearch ---
        results = await store.asimilarity_search(
            query=user_query,
            k=4,
        )
        # LangChain stores text in `page_content`
        long_term_memories = [r.page_content for r in results if r.page_content]

    return {
        "messages": remove_all + [new_user_msg],
        "current_step": 0,
        "plan": None,
        "replan_flag": False,
        "replan_attempts": {},
        "agent_query": None,
        "last_reason": None,
        "user_id": user_id,
        "user_query": user_query,
        "long_term_memories": long_term_memories,
        # Reset token_usage each turn: None triggers accumulate_or_reset → [].
        # node_timings resets via a different path: timed_node overwrites the None with a
        # 'memory_prep' timing record, so the reducer detects that record as a turn boundary.
        "token_usage": None,
    }


#
async def planner_node(state: State) -> Command[Literal["executor"]]:
    """
    Runs the planning LLM and stores the resulting plan in state.
    """
    # 1. Invoke LLM with the planner prompt
    llm_reply = await reasoning_llm.ainvoke([plan_prompt(state)])
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
async def executor_node(
    state: State,
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
    llm_reply = await reasoning_llm.ainvoke([executor_prompt(state)])
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
async def synthesizer_node(state: State) -> Command[Literal["memory_write"]]:
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
        - If the conversation context shows a prior related question, make the answer continuous (e.g. "Compared to April, May had..."). Do not re-introduce yourself.
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

    # Use AWAIT and AINVOKE here with the tracking tag for our event stream
    llm_reply = await reasoning_llm.ainvoke(
        summary_prompt, config={"tags": ["final_synthesis"]}
    )

    answer = llm_reply.content.strip()

    return Command(
        update={
            "messages": [HumanMessage(content=answer, name="synthesizer")],
            "token_usage": [{"node": "synthesizer", **sum_token_usage([llm_reply])}],
        },
        goto="memory_write",
    )


#
async def memory_write_node(state: State) -> dict:
    """
    In-graph node, runs at the end of each turn. ONLY short-term state:
    - Appends this turn to chat_history
    - Summarizes + trims history when it exceeds the threshold
    Long-term fact extraction is NOT done here — see persist_long_term(), scheduled
    by the runner after the response is delivered (Step 5).
    """
    SUMMARY_THRESHOLD = 6  # trim + summarize when chat_history exceeds this many turns
    SUMMARY_PROMPT = """
    You are a memory manager. The following are conversation turns that are being archived.
    Write a concise summary (3-5 sentences) capturing the key questions asked, data retrieved,
    and any user preferences or decisions stated. This summary will be prepended to future prompts.
    Turns to summarize:
    {turns}
    Summary:"""
    WINDOW_K = 3
    user_query = state.get("user_query", "")

    # Find the synthesizer's final answer
    answer = ""
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "name", None) == "synthesizer":
            answer = msg.content
            break

    # 1. Append this turn to chat_history
    new_turns = [
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": answer},
    ]

    chat_history = list(state.get("chat_history") or []) + new_turns
    running_summary = state.get("running_summary") or ""

    # 2. Summarize + trim when history exceeds threshold
    if len(chat_history) > SUMMARY_THRESHOLD * 2:  # *2 because each turn = 2 entries
        turns_to_archive = chat_history[: -WINDOW_K * 2]
        turns_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in turns_to_archive
        )
        summary_input = (
            f"Existing summary:\n{running_summary}\n\nNew turns:\n{turns_text}"
            if running_summary
            else turns_text
        )
        summary_reply = await reasoning_llm.ainvoke(
            [HumanMessage(content=SUMMARY_PROMPT.format(turns=summary_input))]
        )
        running_summary = summary_reply.content.strip()
        chat_history = chat_history[-WINDOW_K * 2 :]  # keep last K turns only

    return {
        "chat_history": chat_history,  # trimmed list replaces prior value (plain Optional, no reducer)
        "running_summary": running_summary,
        "long_term_memories": [],  # clear transient field after use
    }


async def persist_long_term(user_id: str, user_query: str, answer: str) -> None:
    """
    Background task — NOT a graph node, NOT awaited on the user's critical path.
    Extracts durable facts from the turn and writes them to the long-term vector store.
    Fully fault-isolated: any failure is logged and swallowed so it can never fail an
    already-answered turn.
    """
    FACT_EXTRACT_PROMPT = """
    You are a memory manager. Read this conversation turn and extract any durable user preferences,
    domain-specific rules, or facts about the user's work that should be remembered for future conversations.
    Examples: "User always wants pallet counts", "User works in the HGS-UK region", "User prefers line charts".
    Return a JSON array of short strings, one per fact. Return [] if nothing durable was learned.

    User query: {user_query}
    Agent answer: {answer}

    JSON array of facts:"""
    if not (store and user_query and answer):
        return
    try:
        fact_reply = await reasoning_llm.ainvoke(
            [
                HumanMessage(
                    content=FACT_EXTRACT_PROMPT.format(
                        user_query=user_query, answer=answer
                    )
                )
            ]
        )
        try:
            from llmclean import strip_fences

            facts: List[str] = json.loads(strip_fences(fact_reply.content))
        except Exception:
            facts = []

        for fact in facts:
            if fact and isinstance(fact, str):
                # --- FIXED: Use LangChain VectorStore methods ---
                await store.aadd_texts(
                    texts=[fact],
                    metadatas=[
                        {"user_id": user_id}
                    ],  # Save user_id for future filtering
                )
    except Exception as exc:
        # Long-term memory is best-effort: log and drop, never propagate to the user path.
        print(f"[memory] persist_long_term failed (dropped): {exc}")


#
def token_summary(final_state):
    records = final_state.get("token_usage") or []
    print("\n--- Token Usage by Node ---")
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
    """Wraps a node function to record its wall-clock execution time into state.
    Updated to handle both sync and async nodes dynamically, and accept varying arguments."""

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        # --- NEW: Accept *args and **kwargs ---
        async def async_wrapper(*args, **kwargs): 
            start = time.perf_counter()
            # --- NEW: Pass *args and **kwargs down to the actual function ---
            result = await fn(*args, **kwargs) 
            elapsed = time.perf_counter() - start
            print(f"[timing] {name}: {elapsed:.2f}s")
            
            # Note: We assume the function returns a dict or Command.
            if isinstance(result, Command):
                if result.update is None:
                    result.update = {}
                result.update["node_timings"] = [{"node": name, "seconds": elapsed}]
            elif isinstance(result, dict):
                # Handle standard dict returns (like memory_prep_node)
                result["node_timings"] = [{"node": name, "seconds": elapsed}]
                
            return result
        return async_wrapper
    else:
        @functools.wraps(fn)
        # --- NEW: Accept *args and **kwargs ---
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            # --- NEW: Pass *args and **kwargs down to the actual function ---
            result = fn(*args, **kwargs) 
            elapsed = time.perf_counter() - start
            print(f"[timing] {name}: {elapsed:.2f}s")
            
            if isinstance(result, Command):
                if result.update is None:
                    result.update = {}
                result.update["node_timings"] = [{"node": name, "seconds": elapsed}]
            elif isinstance(result, dict):
                 result["node_timings"] = [{"node": name, "seconds": elapsed}]
                 
            return result
        return sync_wrapper


#
def timing_summary(final_state, total_seconds=None):
    timings = final_state.get("node_timings") or []
    print("\n--- Node Execution Times ---")
    for rec in timings:
        print(f"{rec['node']}: {rec['seconds']:.2f}s")
    print(f"Sum of node times: {sum(r['seconds'] for r in timings):.2f}s")
    if total_seconds is not None:
        print(f"Total run time:    {total_seconds:.2f}s")


async def main():
    KEY = os.getenv("cosmos_key")
    ENDPOINT = os.getenv("cosmos_url")
    openai_key = os.getenv("gemini_key")
    azure_openai_key = os.getenv("azure_openai_key")
    embedding_base_url = os.getenv("embedding_base_url")
    embedding_key = os.getenv("embedding_key")
    embedding_deployment = os.getenv("embedding_deployment")
    use_mcp = False
    #
    conversation_id = "conv-abc123"  # from front-end / request body
    user_id = "user-xyz456"  # from auth layer

    # Needs to be global so synthesizer_node can access it
    global reasoning_llm
    reasoning_llm = ChatOpenAI(
        model="DeepSeek-V4-Pro",
        base_url="https://3t-ai-resource.services.ai.azure.com/openai/v1",
        api_key=azure_openai_key,
        max_tokens=2048,
        temperature=0.1,
    )

    #
    if use_mcp:
        mcp_client = MultiServerMCPClient(
            {
                "cosmos": {
                    "command": "python3",
                    "args": [os.path.join(os.path.dirname(__file__), "mcp_server.py")],
                    "transport": "stdio",
                }
            }
        )
        mcp_tools = await mcp_client.get_tools()
        route_tool = next(t for t in mcp_tools if t.name == "query_route_data")
        order_tool = next(t for t in mcp_tools if t.name == "query_order_data")
        cosmos_order_agent = CosmosOrderAgent(llm=reasoning_llm, query_tool=order_tool)
        cosmos_route_agent = CosmosRouteAgent(llm=reasoning_llm, query_tool=route_tool)
    else:
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
    embeddings = OpenAIEmbeddings(
        model=embedding_deployment,
        base_url=f"{embedding_base_url}/openai/v1",
        api_key=embedding_key,
    )
    os.environ["COSMOSDB_ENDPOINT"] = ENDPOINT
    os.environ["COSMOSDB_KEY"] = KEY
    saver = CosmosDBSaver(
        database_name="data-agent", container_name="agent_checkpoints"
    )
    client = CosmosClient(ENDPOINT, credential=KEY)
    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": 1536,  # UPDATE THIS if not using OpenAI embeddings!
            }
        ]
    }
    indexing_policy = {
        "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
    }
    cosmos_container_properties = {"partition_key": PartitionKey(path="/id")}
    vector_search_fields = {"text_field": "text", "embedding_field": "embedding"}
    global store
    store = AzureCosmosDBNoSqlVectorSearch(
        cosmos_client=client,
        database_name="data-agent",
        container_name="agent_memory",
        embedding=embeddings,
        vector_embedding_policy=vector_embedding_policy,
        indexing_policy=indexing_policy,
        cosmos_container_properties=cosmos_container_properties,
        cosmos_database_properties={},
        vector_search_fields=vector_search_fields,
    )
    #
    workflow = StateGraph(State)
    workflow.add_node("memory_prep", timed_node("memory_prep", memory_prep_node))
    workflow.add_node("planner", timed_node("planner", planner_node))
    workflow.add_node("executor", timed_node("executor", executor_node))
    workflow.add_node(
        "cosmos_order", timed_node("cosmos_order", cosmos_order_agent.node)
    )
    workflow.add_node(
        "cosmos_route", timed_node("cosmos_route", cosmos_route_agent.node)
    )
    workflow.add_node("synthesizer", timed_node("synthesizer", synthesizer_node))
    workflow.add_node("memory_write", timed_node("memory_write", memory_write_node))
    workflow.add_edge(START, "memory_prep")
    workflow.add_edge("memory_prep", "planner")
    graph = workflow.compile(checkpointer=saver)
    #
    query = """
    Calculate the average number of routes planned each calendar day for April 2026, do not return only the overall average and follow output example below. Visualize the results by a line chart.
    <Example output from query>
    April 8: 8 routes
    April 10: 15 routes
    <Example output from query>
    """
    # query = """
    # Calculate the average number of routes planned each calendar day for June 2026, and compare with April and May 2026. 
    # """
    # query = """
    # What is the average number of routes planned each calendar day for June 2026.
    # """

    state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "enabled_agents": ["cosmos_route", "cosmos_order", "synthesizer"],
    }
    config = {
        "configurable": {
            "thread_id": conversation_id,
            "user_id": user_id,
        }
    }


    _run_start = time.perf_counter()
    final_state = None

    print("\nProcessing agents... please wait.")
    print("\n--- Final Answer ---\n")

    # Stream events from the graph
    async for event in graph.astream_events(state, config, version="v2"):
        kind = event["event"]
        tags = event.get("tags", [])

        # 1. Stream the Agent steps (UX Improvement)
        if kind == "on_chain_start" and event["name"] in [
            "planner",
            "executor",
            "cosmos_order",
            "cosmos_route",
        ]:
            print(
                f"\n[System: Running {event['name'].replace('_', ' ').title()}...]",
                flush=True,
            )
            # In a real app, yield this as a UI status badge to frontend
        # 2. Stream our final synthesis text chunk by chunk
        elif kind == "on_chat_model_stream" and "final_synthesis" in event.get(
            "tags", []
        ):
            chunk = event["data"]["chunk"]
            print(chunk.content, end="", flush=True)

        # 2. Capture the graph's final state dictionary on completion
        if kind == "on_chain_end" and event["name"] == "LangGraph":
            final_state = event["data"].get("output")

    _total = time.perf_counter() - _run_start
    print("\n\n--------------------------------")

    if final_state:
        token_summary(final_state)
        timing_summary(final_state, _total)

        # Display the chart inline if the cosmos agent produced one
        chart_b64 = final_state.get("chart_b64")
        print("=" * 150)

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

        # long-term vector store: extract + persist durable facts
        answer = ""
        for msg in reversed(final_state.get("messages", [])):
            if getattr(msg, "name", None) == "synthesizer":
                answer = msg.content
                break
        task = asyncio.create_task(
            persist_long_term(user_id, final_state.get("user_query", ""), answer)
        )
        # Keep a reference + log failures so the task isn't garbage-collected or fails silently.
        task.add_done_callback(
            lambda t: t.exception()
            and print(f"[memory] background task error: {t.exception()}")
        )
        # One-shot script: await the write before the event loop closes, or it is cancelled.
        await task
        #
        state_to_save = final_state.copy()
        if "chart_b64" in state_to_save and state_to_save["chart_b64"]:
            state_to_save["chart_b64"] = "[BASE64_IMAGE_DATA_OMITTED]"
        log_path = os.path.join(os.path.dirname(__file__), "output/state_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(state_to_save, f, indent=4, default=str)
    else:
        print("Error: Could not capture final state from graph stream.")

    # Close async Cosmos clients to prevent "Unclosed client session" warnings.
    await cosmos_order_agent.aclose()
    await cosmos_route_agent.aclose()


if __name__ == "__main__":
    # Execute the asynchronous main flow
    asyncio.run(main())
