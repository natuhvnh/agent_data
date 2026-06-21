import os
import re
import json
import glob
import base64
from azure.cosmos import CosmosClient
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.types import Command
from langgraph.graph import END
from prompts import agent_system_prompt, sum_token_usage
from helper import python_repl_tool


class CosmosRouteAgent:
    def __init__(self, llm, query_tool):
        self.llm = llm
        self.query_tool = query_tool
        self.agent = self._create_agent()

    def _get_tools(self):
        return [self.query_tool, python_repl_tool]

    def _create_agent(self):
        """Builds the internal Cosmos data agent."""
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
        """) + " " + " " + query_rule
        return create_agent(self.llm, tools=tools, system_prompt=system_prompt)

    async def node(self, state):
        """
        The graph node function. It invokes the internal agent and
        returns a Command to update the state and route the graph.
        """
        # Snapshot PNGs before invocation so we can detect any new file afterward
        before = {p: os.path.getmtime(p) for p in glob.glob("*.png")}

        # Invoke the agent
        result = await self.agent.ainvoke(state)
        final_msg = HumanMessage(
            content=result["messages"][-1].content, name="cosmos_route"
        )
        usage = sum_token_usage(result["messages"])

        # Detect newly created/modified PNG and base64-encode it for the front-end
        chart_b64 = None
        new = [p for p in glob.glob("*.png")
               if p not in before or os.path.getmtime(p) > before[p]]
        if new:
            latest = max(new, key=os.path.getmtime)
            with open(latest, "rb") as f:
                chart_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()
            os.remove(latest)  # clean up ephemeral disk; remove this line to keep the file

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
    def __init__(self, llm, query_tool):
        self.llm = llm
        self.query_tool = query_tool
        self.agent = self._create_agent()

    def _get_tools(self):
        return [self.query_tool, python_repl_tool]

    def _create_agent(self):
        """Builds the internal Cosmos data agent."""
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
        """) + " " + " " + query_rule
        return create_agent(self.llm, tools=tools, system_prompt=system_prompt)

    async def node(self, state):
        """
        The graph node function. It invokes the internal agent and
        returns a Command to update the state and route the graph.
        """
        # Snapshot PNGs before invocation so we can detect any new file afterward
        before = {p: os.path.getmtime(p) for p in glob.glob("*.png")}

        # Invoke the agent
        result = await self.agent.ainvoke(state)
        final_msg = HumanMessage(
            content=result["messages"][-1].content, name="cosmos_order"
        )
        usage = sum_token_usage(result["messages"])

        # Detect newly created/modified PNG and base64-encode it for the front-end
        chart_b64 = None
        new = [p for p in glob.glob("*.png")
               if p not in before or os.path.getmtime(p) > before[p]]
        if new:
            latest = max(new, key=os.path.getmtime)
            with open(latest, "rb") as f:
                chart_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()
            os.remove(latest)  # clean up ephemeral disk; remove this line to keep the file

        return Command(
            update={
                # emit only the final answer; internal ReAct transcript stays inside the agent
                "messages": [final_msg],
                "chart_b64": chart_b64,
                "token_usage": [{"node": "cosmos_order", **usage}],
            },
            goto="executor"
        )