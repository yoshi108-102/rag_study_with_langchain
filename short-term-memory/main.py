from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from dotenv import load_dotenv
from src import *
load_dotenv()
agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),  
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",  
        "preferences": {"theme": "dark"}  
    },
    {"configurable": {"thread_id": "1"}})

print(result["messages"][-1].content)