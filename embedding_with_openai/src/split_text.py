from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
def split_text(docs: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,add_start_index=True)
    return text_splitter.split_documents(docs)

def main():
    from fetch_docs import fetch_docs
    docs = fetch_docs(["https://lilianweng.github.io/posts/2023-06-23-agent/"])
    chunks = split_text(docs)
    print(chunks[0])

if __name__ == "__main__":
    main()