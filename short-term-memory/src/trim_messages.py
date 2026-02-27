from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any
from dotenv import load_dotenv
load_dotenv()

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 2:
        return None  # No changes needed

    first_msg = messages[1]
    recent_messages = messages[-3:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }
if __name__ == "__main__":
    agent = create_agent(
        model="gpt-5",
        tools=[],
        middleware=[trim_messages],
        checkpointer=InMemorySaver(),
    )

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    agent.invoke({"messages": "hi, my name is yoshihito"}, config)
    agent.invoke({"messages": "write a short poem about cats"}, config)
    agent.invoke({"messages": "now do the same but for dogs"}, config)
    final_response = agent.invoke({"messages": "what's my name?"}, config)
    print(final_response)
    final_response["messages"][-1].pretty_print()
    """
    ================================== Ai Message ==================================

    Your name is Yoshihito. You told me that earlier.
    If you'd like me to call you a nickname or use a different name, just say the word.
    """