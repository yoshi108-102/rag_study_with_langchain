"""
SummarizationMiddleware の要約メッセージのみを表示するスクリプト。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

load_dotenv()


def main():
    checkpointer = InMemorySaver()

    agent = create_agent(
        model="gpt-4.1",
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model="gpt-4.1-mini",
                trigger=("tokens", 500),
                keep=("messages", 4),
            )
        ],
        checkpointer=checkpointer,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "summarization-test"}}

    conversations = [
        "こんにちは！私の名前は太郎です。趣味はプログラミングと登山です。",
        "Pythonについて詳しく教えてください。特にデコレータの仕組みが知りたいです。",
        "ありがとうございます。次に、Pythonのジェネレータについても詳しく教えてください。イテレータとの違いも含めて。",
        "では、非同期プログラミングについても教えてください。asyncioの基本的な使い方と、マルチスレッドとの違いを教えてください。",
        "最後に、私の名前を覚えていますか？最初に言った趣味は何でしたか？",
    ]

    for step, user_msg in enumerate(conversations, 1):
        agent.invoke({"messages": user_msg}, config)

        # チェックポイントから要約メッセージだけ抽出して表示
        state = agent.get_state(config)
        messages = state.values.get("messages", [])
        summary = None
        for msg in messages:
            if "summary" in msg.content.lower():
                summary = msg.content
                break

        if summary:
            print(f"\n--- Step {step} の要約 ---\n{summary}")
        else:
            print(f"\n--- Step {step} ---\n(要約なし)")


if __name__ == "__main__":
    main()
