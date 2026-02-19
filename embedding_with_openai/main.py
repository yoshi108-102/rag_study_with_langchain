from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
import bs4
from src import fetch_docs, split_text, create_retrieve_context_tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

def main():
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
        step["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()
