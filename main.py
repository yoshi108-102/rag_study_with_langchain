def main():
    import os
    from dotenv import load_dotenv
    from langchain_core.vectorstores import InMemoryVectorStore
    load_dotenv()
    
    vector_store = InMemoryVectorStore()
    print(os.getenv("LANGSMITH_TRACING"))
    print(os.getenv("LANGSMITH_API_KEY"))



if __name__ == "__main__":
    main()
