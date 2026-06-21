import os
import re
import json
import glob
import base64
import shutil
# from azure.cosmos import CosmosClient
from azure.cosmos.aio import CosmosClient   # async client
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.types import Command
from langgraph.graph import END
from prompts import agent_system_prompt, sum_token_usage
from helper import python_repl_tool

# Max serialized result payload returned to the LLM (~5k tokens at 4 chars/token)
MAX_RESULT_CHARS = 20_000

# Regex that matches whole-document selects the agent must never issue
_WHOLE_DOC_RE = re.compile(
    r"select\s+(\*|value\s+c\b|c\s+from)",
    re.IGNORECASE,
)


async def run_cosmos_query(container, query: str):
    """Async version — awaits the query iterator."""
    normalized = re.sub(r"\s+", " ", query).strip()
    if _WHOLE_DOC_RE.search(normalized):
        return "Query rejected: ..."
    try:
        results = []
        async for item in container.query_items(query=query):
            results.append(item)
    except Exception as e:
        return f"Query failed: {str(e)}"

    if not results:
        return "No results found."

    serialized = json.dumps(results, default=str)
    if len(serialized) > MAX_RESULT_CHARS:
        return "Query rejected: result too large. ..."
    return results


class CosmosRouteAgent:
    def __init__(
        self, endpoint: str, key: str, database_name: str, container_name: str, llm
    ):
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
        async def query_cosmos_db(query: str):
            """Executes a cosmos query against the Azure Cosmos DB NoSQL container."""
            return await run_cosmos_query(self.container, query)

        return [query_cosmos_db, python_repl_tool]

    def _create_agent(self):
        """Builds the internal Cosmos data agent."""
        schema_context = """
            The container is named 'route'. Use the following schema for SQL generation:

            ### ROOT ATTRIBUTES
            - id: String (Unique record ID)
            - col_date: DateTime string (The collection date of the routing run)
            - run_time: Number (Duration of the optimization process)
            - routes: Array of Route Objects

            ### ROUTE OBJECTS (Nested in c.routes)
            - route_id: Integer (Index of the route)
            - vehicle_name: String (Name of the vehicle e.g., 'Groupage')
            - weight_delivery: Number (Total weight)
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
        query_rule = """
            Cosmos Aggregate Query Rule:
            Whenever COUNT, SUM, AVG, MIN, or MAX is used, generate:
            SELECT VALUE <aggregate>
            instead of:
            SELECT <aggregate>
            because queries are executed with enable_cross_partition_query=True.
        """
        tools = self._get_tools()
        system_prompt = agent_system_prompt("""
            You are the Cosmos DB Expert for a logistics and routing system. When a user asks about route or output from the optimiser data,
            Use the schema context and query rules as reference below and then use 'query_cosmos_db' to fetch the answer.
            Provide a concise summary of the findings with plain text only, do not use markdown.
            While query: You have to use 'VALUE' with the aggregate for cross partition query,
            Order-by over correlated collections is not supported.
            If the user requests a chart or visualization, use 'python_repl_tool' to generate it
            with matplotlib and save it as a .png file in the current working directory. Do not print or log anything.
        """) + " " + schema_context + " " + query_rule
        return create_agent(self.llm, tools=tools, system_prompt=system_prompt)

    async def node(self, state):
        """
        The graph node function. It invokes the internal agent and
        returns a Command to update the state and route the graph.
        """
        # Invoke the agent
        result = await self.agent.ainvoke(state)
        final_msg = HumanMessage(
            content=result["messages"][-1].content, name="cosmos_route"
        )
        usage = sum_token_usage(result["messages"])

        # Detect newly created/modified PNG and base64-encode it for the front-end
        chart_b64 = None
        chart_workdir = None
        for msg in result["messages"]:
            # ToolMessage content from python_repl_tool
            content = getattr(msg, "content", "") or ""
            m = re.search(r"CHART_PATH=(.+?)(\n|$)", content)
            if m:
                chart_path = m.group(1).strip()
                chart_workdir = os.path.dirname(chart_path)
                if os.path.exists(chart_path):
                    with open(chart_path, "rb") as f:
                        chart_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()
                break  # use the last/only chart

        # Clean up the isolated temp dir
        if chart_workdir:
            shutil.rmtree(chart_workdir, ignore_errors=True)

        return Command(
            update={
                # emit only the final answer; internal ReAct transcript stays inside the agent
                "messages": [final_msg],
                "chart_b64": chart_b64,
                "token_usage": [{"node": "cosmos_route", **usage}],
            },
            goto="executor"
        )


class CosmosOrderAgent:
    def __init__(
        self, endpoint: str, key: str, database_name: str, container_name: str, llm
    ):
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
        async def query_cosmos_db(query: str):
            """Executes a cosmos query against the Azure Cosmos DB NoSQL container."""
            return await run_cosmos_query(self.container, query)

        return [query_cosmos_db, python_repl_tool]

    def _create_agent(self):
        """Builds the internal Cosmos data agent."""
        schema_context = """
            The container is named 'route_request'. Use the following schema for SQL generation:
            ### ROOT ATTRIBUTES
            - id: String (Unique request record ID)
            - col_date: DateTime string (Collection date for the routing request to the optimiser)
            - variant: String (Optimization variant/algorithm used)
            - multi_visit_penalty: Boolean (Whether multi-visit penalty is enabled)
            - biggest_equipment: String (Largest available equipment/vehicle type)
            - unit_type: String (Load unit type, e.g., 'Pallet')
            - num_stack: Integer (Maximum stacking level)
            - equipment_list: Array of Equipment Objects
            - order: Array of Order Objects

            ### EQUIPMENT OBJECTS (Nested in c.equipment_list)
            - id: Integer (Equipment identifier)
            - name: String (Equipment name)
            - code: String (Equipment code)
            - internalLengthMillimeter: Integer (Internal vehicle length in mm)
            - internalWidthMillimeter: Integer (Internal vehicle width in mm)
            - internalHeightMillimeter: Integer (Internal vehicle height in mm)
            - maximumPayloadKg: Integer (Maximum payload capacity in kg)
            - palletSpacesUK: Integer (UK pallet capacity)
            - palletSpacesEU: Integer (EU pallet capacity)
            - volume: Number (Vehicle volume capacity)
            - maximumDrivingTimeInMinutes: Integer (Maximum driving time allowed)

            ### ORDER OBJECTS (Nested in c.order)
            - id: String (Unique order identifier)
            - order_number: String (Order reference number)
            - order_line: String (Order line number)
            - accountName: String (Customer account reference)
            - owner_name: String (Owner/customer name)
            - dest_name: String (Delivery location name)
            - stack_on_top: String ('Y' or 'N')
            - stack_on_other: String ('Y' or 'N')

            ### Important Notes:
            - Each document represents a routing request.
            - Orders are stored in the nested array c.order.
            - Equipment specifications are stored in the nested array c.equipment_list.
            - Use JOIN o IN c.order when querying order-level data.
            - Use JOIN e IN c.equipment_list when querying equipment-level data.
            - Aggregate metrics such as destination counts should typically be calculated from the order array.
        """
        query_rule = """
            Cosmos Aggregate Query Rule:
            Whenever COUNT, SUM, AVG, MIN, or MAX is used, generate:
            SELECT VALUE <aggregate>
            instead of:
            SELECT <aggregate>
            because queries are executed with enable_cross_partition_query=True.
        """
        tools = self._get_tools()
        system_prompt = agent_system_prompt("""
            You are the Cosmos DB Expert for a logistics and routing system. When a user asks about the request data used by the optimiser,
            Use the schema context and query rules as reference below and then use 'query_cosmos_db' to fetch the answer.
            Provide a concise summary of the findings with plain text only, do not use markdown.
            While query: You have to use 'VALUE' with the aggregate for cross partition query,
            Order-by over correlated collections is not supported.
            If the user requests a chart or visualization, use 'python_repl_tool' to generate it
            with matplotlib and save it as a .png file in the current working directory. Do not print or log anything
        """) + " " + schema_context + " " + query_rule
        return create_agent(self.llm, tools=tools, system_prompt=system_prompt)

    async def node(self, state):
        """
        The graph node function. It invokes the internal agent and
        returns a Command to update the state and route the graph.
        """
        # Invoke the agent
        result = await self.agent.ainvoke(state)
        final_msg = HumanMessage(
            content=result["messages"][-1].content, name="cosmos_order"
        )
        usage = sum_token_usage(result["messages"])

        # Detect newly created/modified PNG and base64-encode it for the front-end
        chart_b64 = None
        chart_workdir = None
        for msg in result["messages"]:
            # ToolMessage content from python_repl_tool
            content = getattr(msg, "content", "") or ""
            m = re.search(r"CHART_PATH=(.+?)(\n|$)", content)
            if m:
                chart_path = m.group(1).strip()
                chart_workdir = os.path.dirname(chart_path)
                if os.path.exists(chart_path):
                    with open(chart_path, "rb") as f:
                        chart_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()
                break  # use the last/only chart

        # Clean up the isolated temp dir
        if chart_workdir:
            shutil.rmtree(chart_workdir, ignore_errors=True)
        return Command(
            update={
                # emit only the final answer; internal ReAct transcript stays inside the agent
                "messages": [final_msg],
                "chart_b64": chart_b64,
                "token_usage": [{"node": "cosmos_order", **usage}],
            },
            goto="executor"
        )