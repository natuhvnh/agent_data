import os
import json
from typing import Literal, Optional, List, Dict, Any, Type
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
from azure.cosmos import CosmosClient
from cosmos_agent import CosmosDataAgent
from prompts import plan_prompt
from prompts import executor_prompt
from helper import agent_system_prompt, python_repl_tool
#
from dotenv import load_dotenv
load_dotenv()

KEY = os.getenv("cosmos_key")
ENDPOINT = os.getenv("cosmos_url")
openai_key = os.getenv("gemini_key")
azure_openai_key = os.getenv("azure_openai_key")
tavily_key = os.getenv("tavily_key")

#
# Custom State class with specific keys
class State(MessagesState):
    user_query: Optional[str] # The user's original query
    enabled_agents: Optional[List[str]] # Makes our multi-agent system modular on which agents to include
    plan: Optional[List[Dict[int, Dict[str, Any]]]] # Listing the steps in the plan needed to achieve the goal.
    current_step: int # Marking the current step in the plan.
    agent_query: Optional[str] # Inbox note: `agent_query` tells the next agent exactly what to do at the current step.
    last_reason: Optional[str] # Explains the executor’s decision to help maintain continuity and provide traceability.
    replan_flag: Optional[bool] # Set by the executor to indicate that the planner should revise the plan.
    replan_attempts: Optional[Dict[int, Dict[int, int]]] # Replan attempts tracked per step number.
#
reasoning_llm = ChatOpenAI(
    model="gemini-2.5-flash-lite", # Gemini 3.1 Flash Lite
    openai_api_key=openai_key,
    openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    max_tokens=2048,
    temperature=0.1
)

# reasoning_llm = ChatOpenAI(
#     model="DeepSeek-V4-Flash",  # gpt-5.4-mini, DeepSeek-V4-Flash
#     base_url="https://3t-ai-resource.services.ai.azure.com/openai/v1",
#     api_key=azure_openai_key,
#     max_tokens=2048,
#     temperature=0.1
# )
#
def planner_node(state: State) -> Command[Literal['executor']]:
    """
    Runs the planning LLM and stores the resulting plan in state.
    """
    # 1. Invoke LLM with the planner prompt
    llm_reply = reasoning_llm.invoke([plan_prompt(state)])
    output = llm_reply.content
    output = strip_fences(output) # remove markdown json in the output
    # 2. Validate JSON
    try:
        content_str = output if isinstance(output, str) else str(output)
        parsed_plan = json.loads(content_str)
    except json.JSONDecodeError:
        raise ValueError(
            f"Planner returned invalid JSON:\n{output}")

    # 3. Store as current plan only
    replan = state.get("replan_flag", False)
    updated_plan: Dict[str, Any] = parsed_plan
    return Command(
        update={
            "plan": updated_plan,
            "messages": [HumanMessage(
                content=llm_reply.content,
                name="replan" if replan else "initial_plan")],
            "user_query": state.get("user_query", state["messages"][0].content),"current_step": 1 if not replan else state["current_step"],
            # Preserve replan flag so executor runs planned agent once before reconsidering
            "replan_flag": state.get("replan_flag", False),
            "last_reason": "",
            "enabled_agents": state.get("enabled_agents"),
        },
        goto="executor",
    )
#
def executor_node(
    state: State,
    MAX_REPLANS=3
) -> Command[Literal["web_researcher", "chart_generator", "synthesizer", "planner"]]:

    plan: Dict[str, Any] = state.get("plan", {})
    step: int = state.get("current_step", 1)

    # 0) If we *just* replanned, 
    # run the planned agent once before reconsidering.
    if state.get("replan_flag"):
        planned_agent = plan.get(str(step), {}).get("agent")
        return Command(
            update={
                "replan_flag": False,
                "current_step": step + 1,  # advance because we executed the planned agent
            },
            goto=planned_agent,
        )

    # 1) Build prompt & call LLM
    llm_reply = reasoning_llm.invoke([executor_prompt(state)])
    output = llm_reply.content
    output = strip_fences(output) # remove markdown json in the output
    try:
        content_str = output if isinstance(output, str) else str(output)
        parsed = json.loads(content_str)
        replan: bool = parsed["replan"]
        goto: str   = parsed["goto"]
        reason: str = parsed["reason"]
        query: str  = parsed["query"]
    except Exception as exc:
        raise ValueError(f"Invalid executor JSON:\n{llm_reply.content}") from exc

    # Upodate the state
    updates: Dict[str, Any] = {
        "messages": [HumanMessage(content=llm_reply.content, name="executor")],
        "last_reason": reason,
        "agent_query": query,
    }

    # Replan accounting
    replans: Dict[int, int] = state.get("replan_attempts", {}) or {}
    step_replans = replans.get(step, 0)

    # 2) Replan decision
    if replan:
        if step_replans < MAX_REPLANS:
            replans[step] = step_replans + 1
            updates.update({
                "replan_attempts": replans,
                "replan_flag": True,     # ensure next turn executes the planned agent once
                "current_step": step,    # stay on same step for the new plan
            })
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
wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
tavily_tool = TavilySearchResults(api_wrapper=wrapper, max_results=5)
web_search_agent = create_agent(
    reasoning_llm,
    tools=[tavily_tool],
    system_prompt=agent_system_prompt(f"""
        You are the Researcher. You can ONLY perform research by using the provided search tool (tavily_tool). 
        When you have found the necessary information, end your output.  
        Do NOT attempt to take further actions.
    """),
)
#
def web_research_node(state: State,) -> Command[Literal["executor"]]:
    agent_query = state.get("agent_query")
    result = web_search_agent.invoke({"messages":agent_query})
    goto = "executor"
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="web_researcher")
    return Command(
        update={
            # share internal message history of research agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )
#
chart_agent = create_agent(reasoning_llm,
                                 tools=[python_repl_tool],
                                 system_prompt=agent_system_prompt(
        """
        You can only generate charts. You are working with a researcher colleague.
        1) Print the chart first.
        2) Save the chart to a file in the current working directory.
        3) At the very end of your message, output EXACTLY two lines so the summarizer can find them:
           CHART_PATH: <relative_path_to_chart_file>
           CHART_NOTES: <one concise sentence summarizing the main insight in the chart>
        Do not include any other trailing text after these two lines.
        """
    ),
)
#
def chart_node(state: State) -> Command[Literal["chart_summarizer"]]:
    result = chart_agent.invoke(state)
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="chart_generator")
    goto="chart_summarizer"
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )
#
chart_summary_agent = create_agent(reasoning_llm,
                                         tools=[],  # Add image processing tools if available/needed.
                                         system_prompt=agent_system_prompt(
     """
     You can only generate image captions. You are working with a researcher colleague and a chart generator colleague.
     Your task is to generate a standalone, concise summary for the provided chart image saved at a local PATH,
     where the PATH should be and only be provided by your chart generator colleague.
     The summary should be no more than 3 sentences and should not mention the chart itself."
     """
        
    ),
)
#
def chart_summary_node(state: State) -> Command[Literal[END]]:
    result = chart_summary_agent.invoke(state)
    print(f"Chart summarizer answer: {result['messages'][-1].content}")
    # Send to the end node
    goto = END
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
            "final_answer": result["messages"][-1].content,
        },
        goto=goto,
    )
#
def synthesizer_node(state: State) -> Command[Literal[END]]:
    """
    Creates a concise, human‑readable summary of the entire interaction, **purely in prose**.
    It ignores structured tables or chart IDs and instead rewrites the relevant agent messages (research results, chart commentary, etc.) into a short final answer.
    """
    # Gather informative messages for final synthesis
    relevant_msgs = [
        m.content for m in state.get("messages", [])
        if getattr(m, "name", None) in ("web_researcher", 
                                        "chart_generator", 
                                        "chart_summarizer")
    ]
    print("="*100)
    print(json.dumps(state, indent=4))

    user_question = state.get("user_query", state.get("messages", [{}])[0].content if state.get("messages") else "")
    synthesis_instructions = (
        """
        You are the Synthesizer. Use the context below to directly answer the user's question.
        Perform any lightweight calculations, comparisons, or inferences required.
        Do not invent facts not supported by the context.
        If data is missing, say what's missing and, if helpful, offer a clearly labeled best-effort estimate with assumptions.
        Produce a concise response that fully answers the question, with 
        the following guidance:
        - Start with the direct answer (one short paragraph or a tight bullet list).\n
        - Include key figures from any 'Results:' tables (e.g., totals, top items).\n
        - If any message contains citations, include them as a brief 'Citations: [...]' line.\n
        - Keep the output crisp; avoid meta commentary or tool instructions.
        """
        )
    summary_prompt = [
        HumanMessage(content=(
            f"User question: {user_question}\n\n"
            f"{synthesis_instructions}\n\n"
            f"Context:\n\n" + "\n\n---\n\n".join(relevant_msgs)
        ))
    ]

    llm_reply = reasoning_llm.invoke(summary_prompt)

    answer = llm_reply.content.strip()
    print(f"Synthesizer answer: {answer}")

    return Command(
        update={
            "final_answer": answer,
            "messages": [HumanMessage(content=answer, name="synthesizer")],
        },
        goto=END,           # hand off to the END node
    )
#
cosmos_agent_instance = CosmosDataAgent(
    endpoint=ENDPOINT,
    key=KEY,
    database_name="hgs-output",
    container_name="route",
    llm=reasoning_llm
)
#
def token_summary(final_state):
    total_input_tokens = 0
    total_output_tokens = 0
    for msg in final_state["messages"]:
        # Check if the message has usage metadata (standard in newer LangChain)
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            total_input_tokens += msg.usage_metadata.get("input_tokens", 0)
            total_output_tokens += msg.usage_metadata.get("output_tokens", 0)
        # Fallback for older versions or specific providers using response_metadata
        elif hasattr(msg, "response_metadata") and "token_usage" in msg.response_metadata:
            usage = msg.response_metadata["token_usage"]
            total_input_tokens += usage.get("prompt_tokens", 0)
            total_output_tokens += usage.get("completion_tokens", 0)
    print(f"--- Total Token Usage ---")
    print(f"Input Tokens:  {total_input_tokens}")
    print(f"Output Tokens: {total_output_tokens}")
    print(f"Total Tokens:  {total_input_tokens + total_output_tokens}")
    return
#
workflow = StateGraph(State)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("cosmos_data", cosmos_agent_instance.node)
workflow.add_node("web_researcher", web_research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("chart_summarizer", chart_summary_node)
workflow.add_node("synthesizer", synthesizer_node)
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
state = {
    "messages": [HumanMessage(content=query)],
    "user_query": query,
    "enabled_agents": ["web_researcher", "chart_generator", "chart_summarizer", "synthesizer", "cosmos_data"],
        }
final_state = graph.invoke(state)
final_answer = final_state["messages"][-1].content
print(final_answer)
print("--------------------------------")
print(final_state)
token_summary(final_state)
