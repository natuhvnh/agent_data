# npx @modelcontextprotocol/inspector python3 mcp_server.py 
import os
import re
import json
from mcp.server.fastmcp import FastMCP #
from azure.cosmos import CosmosClient
from dotenv import load_dotenv
load_dotenv()


# Max serialized result payload returned to the LLM (~5k tokens at 4 chars/token)
MAX_RESULT_CHARS = 20_000

# Regex that matches whole-document selects the agent must never issue
_WHOLE_DOC_RE = re.compile(
    r"select\s+(\*|value\s+c\b|c\s+from)",
    re.IGNORECASE,
)


def run_cosmos_query(container, query: str):
    """
    Execute a Cosmos DB SQL query with two enforcement guards:
      1. Reject whole-document selects (SELECT *, SELECT c, SELECT VALUE c).
      2. Reject results whose serialized payload exceeds MAX_RESULT_CHARS.
    Returns the result list, or an error string the agent can act on.
    """
    normalized = re.sub(r"\s+", " ", query).strip()
    if _WHOLE_DOC_RE.search(normalized):
        return (
            "Query rejected: whole-document selects are not allowed. "
            "Project only the specific scalar fields you need "
            "(e.g. SELECT c.id, c.col_date, ARRAY_LENGTH(c.routes)) "
            "or use an aggregate (COUNT, SUM, AVG)."
        )
    try:
        # print(query)
        results = list(
            container.query_items(query=query, enable_cross_partition_query=True)
        )
    except Exception as e:
        return f"Query failed: {str(e)}"

    if not results:
        return "No results found."

    serialized = json.dumps(results, default=str)
    if len(serialized) > MAX_RESULT_CHARS:
        return (
            f"Query rejected: result payload is {len(serialized):,} chars "
            f"(limit {MAX_RESULT_CHARS:,}). "
            "Add an aggregate (COUNT, SUM, AVG) or project fewer/shorter fields."
        )
    return results


mcp = FastMCP("Cosmos-Database-Server")
_client = CosmosClient(os.getenv("cosmos_url"), os.getenv("cosmos_key"))
_route_container = (
    _client.get_database_client("hgs-output").get_container_client("route")
)
_order_container = (
    _client.get_database_client("hgs-input").get_container_client("orders")
)


@mcp.tool()
def query_route_data(query: str) -> str:
    """Run a read-only Azure Cosmos NoSQL query against optimizer OUTPUT data
    (database: hgs-output, container: route).

    SCHEMA
    ------
    ROOT ATTRIBUTES
    - id: String — unique record ID
    - col_date: DateTime string — collection date of the routing run
    - run_time: Number — duration of the optimization process
    - routes: Array of Route Objects

    ROUTE OBJECTS (c.routes[])
    - route_id: Integer — index of the route
    - vehicle_name: String — e.g. 'Groupage'
    - weight_delivery / volume_delivery / pallet_delivery: Number
    - distance: Number — total distance for this route
    - vehicle_weight_utilization / vehicle_volume_utilization / vehicle_pallet_utilization: Float (0.0–1.0)
    - service_time: Number — total time in minutes
    - visits: Array of Visit Objects

    VISIT OBJECTS (c.routes[].visits[])
    - visit_name: String — delivery location/customer name
    - order_number: Array — e.g. ["ARM0000700"]
    - orderIds: Array of UUIDs

    QUERY RULES
    -----------
    - Project specific fields only. Whole-document selects (SELECT *, SELECT VALUE c,
      SELECT c FROM c) are rejected.
    - For cross-partition aggregates use: SELECT VALUE <agg>
      e.g. SELECT VALUE COUNT(1) FROM c WHERE c.col_date >= '2026-04-01'
    - ORDER BY over correlated sub-collections is not supported.
    """
    result = run_cosmos_query(_route_container, query)
    return json.dumps(result, default=str) if isinstance(result, list) else str(result)


@mcp.tool()
def query_order_data(query: str) -> str:
    """Run a read-only Azure Cosmos NoSQL query against optimizer INPUT data
    (database: hgs-input, container: orders).

    SCHEMA
    ------
    ROOT ATTRIBUTES
    - id: String — unique request record ID
    - col_date: DateTime string — collection date for the routing request
    - variant: String — optimization variant/algorithm used
    - multi_visit_penalty: Boolean
    - biggest_equipment: String — largest available equipment/vehicle type
    - unit_type: String — e.g. 'Pallet'
    - num_stack: Integer — maximum stacking level
    - equipment_list: Array of Equipment Objects
    - order: Array of Order Objects

    EQUIPMENT OBJECTS (c.equipment_list[])
    - id / name / code: Integer / String / String
    - internalLengthMillimeter / internalWidthMillimeter / internalHeightMillimeter: Integer
    - maximumPayloadKg: Integer
    - palletSpacesUK / palletSpacesEU: Integer
    - volume: Number
    - maximumDrivingTimeInMinutes: Integer

    ORDER OBJECTS (c.order[])
    - id: String — unique order identifier
    - order_number / order_line: String
    - accountName / owner_name: String — customer references
    - dest_name: String — delivery location name
    - stack_on_top / stack_on_other: 'Y' or 'N'

    QUERY RULES
    -----------
    - Project specific fields only. Whole-document selects are rejected.
    - For cross-partition aggregates use: SELECT VALUE <agg>
    - Use JOIN o IN c.order when querying order-level data.
    - Use JOIN e IN c.equipment_list when querying equipment-level data.
    - ORDER BY over correlated sub-collections is not supported.
    """
    result = run_cosmos_query(_order_container, query)
    return json.dumps(result, default=str) if isinstance(result, list) else str(result)


if __name__ == "__main__":
    # Run the server using stdio transport (local subprocess communication)
    mcp.run(transport="stdio") #