from langchain_core.vectorstores import VectorStore
from langchain_core.tools import tool  
from langchain.agents.middleware import dynamic_prompt, ModelRequest

def create_retrieve_context_tool(vector_store: VectorStore):
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    return retrieve_context

def create_retrieve_context_with_dynamic_prompt(vector_store: VectorStore, prompt: str):
    @dynamic_prompt
    def prompt_with_context(request: ModelRequest):
        """Inject context into state messages."""
        last_query = request.state["messages"][-1].text
        retrieved_docs = vector_store.similarity_search(last_query)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        system_message = (
            f"{prompt}\n\n{docs_content}"
        )

        return system_message
    return prompt_with_context