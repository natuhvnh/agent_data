from langgraph.graph import MessagesState
from typing import Annotated, Optional, List, Dict, Any


def accumulate_or_reset(current: Optional[list], update: Optional[list]) -> list:
    """Append `update` to `current`, but reset to a fresh list at the start of each turn.

    Reset triggers on EITHER:
    - an explicit None write  — memory_prep uses this for token_usage, OR
    - a 'memory_prep' record  — for node_timings: timed_node wraps memory_prep_node and
      overwrites its None with a timing dict, so we detect the turn-start node instead.

    This keeps both metric channels strictly per-turn even on a long-running conversation
    thread where the checkpointer would otherwise accumulate across turns.
    """
    if update is None:
        return []
    if any(isinstance(r, dict) and r.get("node") == "memory_prep" for r in update):
        return list(update)   # new turn → discard prior turn's records, start fresh
    return (current or []) + list(update)


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
        List[Dict[str, Any]], accumulate_or_reset
    ]  # per-turn wall-time records (memory_prep resets each turn via None)
    token_usage: Annotated[
        List[Dict[str, Any]], accumulate_or_reset
    ]  # per-turn token counts (memory_prep resets each turn via None)
    chat_history: Optional[List[Dict[str, Any]]]
    running_summary: Optional[
        str
    ]  # LLM-compressed summary of older turns that have been trimmed from chat_history.
    long_term_memories: Optional[
        List[str]
    ]  # Facts retrieved from the vector store for THIS turn only (transient; not persisted across turns)
    user_id: Optional[str]  # Set by memory_prep from config; used by memory_write