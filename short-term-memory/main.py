from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from dotenv import load_dotenv
from src import *
import json

load_dotenv()

# ─── ヘルパー: チェックポイントの中身を可視化 ───
def show_checkpoint(checkpointer: InMemorySaver, thread_id: str, turn: int):
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"📸 Turn {turn} 後の InMemorySaver の状態  (thread_id={thread_id!r})")
    print(separator)

    # storage 内部を覗く
    storage = checkpointer.storage
    print(f"\n🗂  保存されている thread 数: {len(storage)}")

    for key, data in storage.items():
        # key は (thread_id, checkpoint_ns, checkpoint_id) のタプル
        print(f"\n  🔑 key: {key}")

    # 最新のチェックポイントを取得
    config = {"configurable": {"thread_id": thread_id}}
    latest = checkpointer.get(config)

    if latest is None:
        print("  (チェックポイントなし)")
        return

    print(f"\n📋 最新チェックポイント:")
    print(f"  checkpoint_id : {latest['id']}")
    print(f"  channel_values keys: {list(latest['channel_values'].keys())}")

    # messages の中身を見る
    messages = latest["channel_values"].get("messages", [])
    print(f"\n💬 メッセージ履歴 ({len(messages)} 件):")
    for i, msg in enumerate(messages):
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))
        # tool の場合は短縮
        preview = content[:120] + ("..." if len(content) > 120 else "")
        print(f"  [{i}] {role:>10}: {preview}")

    # カスタム state のフィールドを見る
    for field in ["user_id", "preferences"]:
        val = latest["channel_values"].get(field)
        if val is not None:
            print(f"\n🏷  {field}: {val}")

    print(f"\n{separator}\n")


# ─── メイン ───
checkpointer = InMemorySaver()

agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=checkpointer,
)

thread_id = "1"
config = {"configurable": {"thread_id": thread_id}}

# ── 会話シナリオ (同じ thread_id で複数ターン) ──
conversations = [
    "Hello, who am I?",
    "What theme do I prefer?",
    "Thanks! Can you remember what we talked about?",
]

for turn, user_msg in enumerate(conversations, 1):
    print(f"\n{'🟢' * 20}")
    print(f"👤 User (Turn {turn}): {user_msg}")
    print(f"{'🟢' * 20}")

    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": user_msg}],
            "user_id": "user_123",
            "preferences": {"theme": "dark"},
        },
        config,
    )

    print(f"🤖 Assistant: {result['messages'][-1].content}")

    # ターンごとにチェックポイントを可視化
    show_checkpoint(checkpointer, thread_id, turn)


# ── おまけ: 別の thread_id で会話して、thread 間の独立性を確認 ──
print("\n" + "🔵" * 30)
print("📌 別の thread (thread_id='2') で会話してみる")
print("🔵" * 30)

result2 = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hi, who am I?"}],
        "user_id": "user_456",
        "preferences": {"theme": "light"},
    },
    {"configurable": {"thread_id": "2"}},
)
print(f"🤖 Assistant: {result2['messages'][-1].content}")

print("\n" + "=" * 70)
print("📊 最終的な InMemorySaver の全体像")
print("=" * 70)
print(f"  保存 thread 数: {len(checkpointer.storage)} エントリ")
print(f"  thread 一覧:")
thread_ids = {k[0] for k in checkpointer.storage.keys()}
for tid in sorted(thread_ids):
    entries = [k for k in checkpointer.storage.keys() if k[0] == tid]
    print(f"    thread_id={tid!r} → チェックポイント {len(entries)} 個")