from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from .get_user_info import get_user_info

class CustomAgentState(AgentState):  
    user_id: str
    preferences: dict
