from __future__ import annotations
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportIncompatibleMethodOverride=false
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Valid config keys have changed in V2",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"WARNING! response_format is not default parameter",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"^munch$",
)

import os
import json
import re
from dotenv import load_dotenv
from snowflake.snowpark import Session
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing import Annotated, Literal, Optional, List, Dict, Any, Type
from trulens.otel.semconv.trace import SpanAttributes
from trulens.core.otel.instrument import instrument
from snowflake.core import Root
from snowflake.core.cortex.lite_agent_service import AgentRunRequest
from pydantic import BaseModel, PrivateAttr
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector
from trulens.providers.openai import OpenAI
import numpy as np
from prompts import plan_prompt, executor_prompt, agent_system_prompt



os.environ["TRULENS_OTEL_TRACING"] = "1"

# load full dotenv
load_dotenv()
repl = PythonREPL()
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. You will be used to execute python code
    that generates charts. Only print the chart once.
    This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = (
        f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    )
    return (
        result_str
        # + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

from IPython.display import HTML, display

def display_eval_reason(text, width=800):
    # Strip any trailing "Score: X" from the end of the text
    raw_text = str(text).rstrip()
    cleaned_text = re.sub(r"\s*Score:\s*-?\d+(?:\.\d+)?\s*$", "", raw_text, flags=re.IGNORECASE)
    # Convert newlines to HTML line breaks, then wrap
    html_text = cleaned_text.replace('\n', '<br><br>')
    display(HTML(f'<div style="font-size: 15px; word-wrap: break-word; width: {width}px;">{html_text}</div>'))