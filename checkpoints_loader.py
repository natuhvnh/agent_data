import os
from langgraph_checkpoint_cosmosdb import CosmosDBSaver
from langgraph.graph import StateGraph, START
from typing import Annotated, Optional, List, Dict, Any
import operator
from state_message import State
from dotenv import load_dotenv

load_dotenv()
# 1. Provide your Cosmos DB credentials
KEY = os.getenv("cosmos_key")
ENDPOINT = os.getenv("cosmos_url")

os.environ["COSMOSDB_ENDPOINT"] = ENDPOINT
os.environ["COSMOSDB_KEY"] = KEY

# 2. Create a minimal "Dummy" Graph
# We don't add any nodes! We just need LangGraph's engine to read the DB.
def dummy_node(state: State):
    # This node does absolutely nothing, it just passes the state through
    return state

workflow = StateGraph(State)
workflow.add_node("dummy", dummy_node)
workflow.add_edge(START, "dummy")
dummy_graph = workflow.compile() # Compile an empty graph

# 3. Initialize the Saver and pull the data
def get_thread_data(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    # --- FIXED: Remove the 'with' statement ---
    saver = CosmosDBSaver(
        database_name="data-agent",
        container_name="agent_checkpoints"
    )
        
    # Attach the saver temporarily to our dummy graph
    dummy_graph.checkpointer = saver
    
    # Pull the latest state dictionary
    try:
        state = dummy_graph.get_state(config)
        return state.values
    except Exception as e:
        print(f"Could not retrieve state for {thread_id}: {e}")
        return None

# --- Usage Example ---
if __name__ == "__main__":
    target_thread = "conv-abc123"
    
    print(f"Fetching data for thread: {target_thread}...\n")
    data = get_thread_data(target_thread)
    
    if data:
        print("--- Chat History ---")
        for turn in data.get("chat_history", []):
            print(f"{turn['role'].upper()}: {turn['content']}")
            
        print("\n--- Final User Query ---")
        print(data.get("user_query"))
        
        print("\n--- Raw Messages Array ---")
        print(f"Total messages in state: {len(data.get('messages', []))}")
    else:
        print("No data found for this thread.")