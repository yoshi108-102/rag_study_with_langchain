from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from dotenv import load_dotenv
load_dotenv()
@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """Look up user info."""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"
