from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
load_dotenv()
if __name__ == "__main__":

    checkpointer = InMemorySaver()

    agent = create_agent(
        model="gpt-4.1",
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model="gpt-4.1-mini",
                trigger=("tokens", 4000),
                keep=("messages", 20)
            )
        ],
        checkpointer=checkpointer,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    agent.invoke({"messages": "hi, my name is bob"}, config)
    agent.invoke({"messages": "write a short poem about cats"}, config)
    agent.invoke({"messages": "now do the same but for dogs"}, config)
    final_response = agent.invoke({"messages": "what's my name?"}, config)

    final_response["messages"][-1].pretty_print()
    print(checkpointer)
    """
    ================================== Ai Message ==================================

    Your name is Bob!
    """