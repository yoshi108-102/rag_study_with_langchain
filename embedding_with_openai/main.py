from typing import Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
import bs4
from src import fetch_docs, split_text, create_retrieve_context_tool,create_retrieve_context_with_dynamic_prompt
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime


load_dotenv()
def tool_rag():
    # Load the document
    docs = fetch_docs(["https://lilianweng.github.io/posts/2023-06-23-agent/"])
    chunks = split_text(docs)
    
    embedding = OpenAIEmbeddings()
    vector_store = InMemoryVectorStore(embedding=embedding)
    document_ids = vector_store.add_documents(chunks)
    
    retrieve_context = create_retrieve_context_tool(vector_store)
    tools = [retrieve_context]
    
    prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
    )
    agent = create_agent(model=ChatOpenAI(), tools=tools, system_prompt=prompt)

    query = "What is task decomposition?"
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        print(step)
        step["messages"][-1].pretty_print()

def dynamic_prompt_rag():
    # Load the document
    docs = fetch_docs(["https://lilianweng.github.io/posts/2023-06-23-agent/"])
    chunks = split_text(docs)
    
    embedding = OpenAIEmbeddings()
    vector_store = InMemoryVectorStore(embedding=embedding)
    
    prompt = "You are a helpful assistant. Use the following context in your response:"
    retrieve_context = create_retrieve_context_with_dynamic_prompt(vector_store,prompt)
    
    
    agent = create_agent(model=ChatOpenAI(), tools=[],middleware=[retrieve_context])

    query = "What is task decomposition?"
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        print(step)
        step["messages"][-1].pretty_print()


class State(AgentState):
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def __init__(self, vector_store: InMemoryVectorStore):
        self.vector_store = vector_store

    def before_model(self, state: AgentState,runtime: Runtime[None]) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        retrieved_docs = self.vector_store.similarity_search(last_message.text)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        augmented_message_content = (
            f"{last_message.text}\n\n"
            "Use the following context to answer the query:\n"
            f"{docs_content}"
        )
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
        }


def agent_middleware_rag():
    """AgentMiddleware を使った RAG のテスト"""
    # ドキュメント読み込み & 分割
    docs = fetch_docs(["https://lilianweng.github.io/posts/2023-06-23-agent/"])
    chunks = split_text(docs)

    # ベクトルストア作成
    embedding = OpenAIEmbeddings()
    vector_store = InMemoryVectorStore(embedding=embedding)
    vector_store.add_documents(chunks)

    # ミドルウェアを使ってエージェント作成
    middleware = RetrieveDocumentsMiddleware(vector_store)
    agent = create_agent(
        model=ChatOpenAI(),
        tools=[],
        middleware=[middleware],
    )

    query = "What is task decomposition?"

    # stream で最終ステートを取得
    final_state = None
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        final_state = step
        step["messages"][-1].pretty_print()

    # ---- raw data の活用デモ ----
    print("\n" + "=" * 60)
    print("📦 ステートに保存された raw documents (context)")
    print("=" * 60)

    raw_docs: list[Document] = final_state.get("context", [])
    print(f"\n取得ドキュメント数: {len(raw_docs)}\n")

    for i, doc in enumerate(raw_docs, 1):
        print(f"--- Document {i} ---")
        print(f"  metadata : {doc.metadata}")
        print(f"  content  : {doc.page_content[:200]}...")
        print()

    # 例: raw data を使った二次加工
    # メッセージには要約済みテキストが入っているが、
    # context にはオリジナルのドキュメントがそのまま残っているので
    # ソース URL の一覧を出したり、別の処理に再利用できる
    print("=" * 60)
    print("🔗 ソース一覧 (metadata から抽出)")
    print("=" * 60)
    sources = {doc.metadata.get("source", "unknown") for doc in raw_docs}
    for src in sources:
        print(f"  - {src}")


if __name__ == "__main__":
    agent_middleware_rag()
