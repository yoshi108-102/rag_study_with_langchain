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


@tool
def inspect_runtime(
    runtime: ToolRuntime
) -> str:
    """Inspect what ToolRuntime can see — state, config, and message history."""
    lines = []

    # 1) config から thread_id を取得
    thread_id = runtime.config.get("configurable", {}).get("thread_id", "unknown")
    lines.append(f"🧵 thread_id: {thread_id}")

    # 2) state のカスタムフィールドにアクセス
    user_id = runtime.state.get("user_id", "N/A")
    preferences = runtime.state.get("preferences", {})
    lines.append(f"👤 user_id: {user_id}")
    lines.append(f"⚙️  preferences: {preferences}")

    # 3) state に保存されているメッセージ履歴を確認
    messages = runtime.state.get("messages", [])
    lines.append(f"💬 messages count: {len(messages)}")
    for i, msg in enumerate(messages):
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))
        preview = content[:80] + ("..." if len(content) > 80 else "")
        lines.append(f"   [{i}] {role}: {preview}")

    return "\n".join(lines)
