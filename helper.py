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
import re
import glob
import shutil
import tempfile
import asyncio
import subprocess
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing import Annotated, Literal, Optional, List, Dict, Any, Type
from pydantic import BaseModel, PrivateAttr
os.environ["TRULENS_OTEL_TRACING"] = "1"

# load full dotenv
load_dotenv()
repl = PythonREPL()

# @tool
# def python_repl_tool(
#     code: Annotated[str, "The python code to execute to generate your chart."],
# ):
#     """Use this to execute python code. You will be used to execute python code that generates and display charts.
#     If using matplotlib to display chart, never call 'plt.show()' or show image to user.
#     """
#     # print("Executing code via subprocess:\n", code)
#     # Save the agent's code to a temporary file
#     with open("temp_agent_script.py", "w") as f:
#         f.write(code)
#     try:
#         # Run the script as a completely separate process
#         result = subprocess.run(
#             ["python3", "temp_agent_script.py"], 
#             capture_output=True, 
#             text=True,
#             check=True
#         )
#         output = result.stdout
#     except subprocess.CalledProcessError as e:
#         return f"Failed to execute. Error: {e.stderr}"
    
#     return f"Successfully executed. Stdout: {output}"

@tool
async def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Execute Python code to generate a matplotlib chart and save it as a .png file.
    Never call plt.show() or display the image inline. Do not print or log anything."""
    workdir = tempfile.mkdtemp(prefix="chart_")
    script_path = os.path.join(workdir, "script.py")
    with open(script_path, "w") as f:
        f.write(code)
    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            shutil.rmtree(workdir, ignore_errors=True)
            return f"Failed to execute. Error: {stderr.decode()}"
        output = stdout.decode()
    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        return f"Failed to execute. Error: {str(e)}"

    pngs = glob.glob(os.path.join(workdir, "*.png"))
    if pngs:
        chart_path = max(pngs, key=os.path.getmtime)
        return f"Successfully executed. CHART_PATH={chart_path}\nStdout: {output}"

    shutil.rmtree(workdir, ignore_errors=True)
    return f"Successfully executed. Stdout: {output}"


from IPython.display import HTML, display
def display_eval_reason(text, width=800):
    # Strip any trailing "Score: X" from the end of the text
    raw_text = str(text).rstrip()
    cleaned_text = re.sub(r"\s*Score:\s*-?\d+(?:\.\d+)?\s*$", "", raw_text, flags=re.IGNORECASE)
    # Convert newlines to HTML line breaks, then wrap
    html_text = cleaned_text.replace('\n', '<br><br>')
    display(HTML(f'<div style="font-size: 15px; word-wrap: break-word; width: {width}px;">{html_text}</div>'))