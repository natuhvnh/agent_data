from azure.cosmos import CosmosClient
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.types import Command
from langgraph.graph import END
from helper import agent_system_prompt, python_repl_tool


class CosmosDataAgent:
    def __init__(self, endpoint: str, key: str, database_name: str, container_name: str, llm):
        """
        Initializes the Cosmos DB connection and the specialized ReAct agent.
        """
        self.client = CosmosClient(endpoint, key)
        self.database = self.client.get_database_client(database_name)
        self.container = self.database.get_container_client(container_name)
        self.llm = llm
        
        # Initialize the agent with bound tools
        self.agent = self._create_agent()
    
    def _get_tools(self):
        """
        Defines tools within the class scope to maintain access to self.container.
        """
        @tool
        def query_cosmos_db(query: str):
            """Executes a SQL query against the Azure Cosmos DB NoSQL container."""
            # print(query)
            # print("="*100)
            try:
                results = list(self.container.query_items(
                    query=query, 
                    enable_cross_partition_query=True
                ))
                return results if results else "No results found."
            except Exception as e:
                return f"Query failed: {str(e)}"
        return [query_cosmos_db, python_repl_tool]

    def _create_agent(self):
        """Builds the internal ReAct agent."""
        schema_context = """
            The container is named 'route'. Use the following schema for SQL generation:

            ### ROOT ATTRIBUTES
            - id: String (Unique record ID)
            - col_date: DateTime string (The collection date of the routing run)
            - run_time: Number (Duration of the optimization process)
            - routes: Array of Route Objects

            ### ROUTE OBJECTS (Nested in c.routes)
            - route_id: Integer (Index of the route)
            - vehicle_name: String (e.g., 'Groupage')
            - weight_delivery: Number (Total weight on route)
            - volume_delivery: Number (Total volume)
            - pallet_delivery: Number (Total pallets)
            - distance: Number (Total distance for this route)
            - vehicle_weight_utilization: Float (0.0 to 1.0)
            - vehicle_volume_utilization: Float (0.0 to 1.0)
            - vehicle_pallet_utilization: Float (0.0 to 1.0)
            - service_time: Number (Total time in minutes)
            - visits: Array of Visit Objects

            ### VISIT OBJECTS (Nested in c.routes[].visits)
            - visit_name: String (Name of the delivery location/customer)
            - order_number: Array (List of ARM numbers, e.g., ["ARM0000700"])
            - orderIds: Array of UUIDs
        """
        tools = self._get_tools()
        system_prompt = agent_system_prompt(
            "You are the Cosmos DB Expert for a logistics and routing system. When a user asks about data, "
            "Use the schema context as reference and then use 'query_cosmos_db' to fetch the answer. "
            "Provide a concise summary of the findings."
        ) + ' ' + schema_context
        return create_agent(self.llm, tools=tools, system_prompt=system_prompt)

    def node(self, state):
        """
        The graph node function. It invokes the internal agent and 
        returns a Command to update the state and route the graph.
        """
        # Invoke the agent
        result = self.agent.invoke(state)
        result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="cosmos_data")
        return Command(
            update={
                # share internal message history of research agent with other agents
                "messages": result["messages"],
            },
            goto=END,
        )