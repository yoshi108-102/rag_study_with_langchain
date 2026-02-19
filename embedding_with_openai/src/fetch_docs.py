from langchain_community.document_loaders import WebBaseLoader
import bs4
def fetch_docs(path: list[str]):
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=path,
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    return docs

def main():
    docs = fetch_docs(["https://lilianweng.github.io/posts/2023-06-23-agent/"])
    assert len(docs) == 1
    print(f"Total characters: {len(docs[0].page_content)}") 
    print(docs[0].page_content[:500])

if __name__ == "__main__":
    main()