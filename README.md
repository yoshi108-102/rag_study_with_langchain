# RAG Study with LangChain

This project is a practical study of Retrieval-Augmented Generation (RAG) using the LangChain framework. It demonstrates how to build a RAG pipeline to answer questions about a specific document.

## Prerequisites

- Python 3.8+
- An OpenAI API key

## Setup

1.  **Clone the repository** (if you haven't already).

2.  **Create a virtual environment** (recommended):
    ```bash
    uv init
    uv sync
    uv pip install -U "langchain[openai]"
    ```

3.  **Install dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    Create a `.env` file in the root directory with the following content:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    LANGSMITH_TRACING="true"
    LANGSMITH_API_KEY="your_langsmith_api_key_here"
    ```

## Usage

### 1. Download Sample Data

Download the `state_of_the_union.txt` file to the `data` directory:

```bash
mkdir -p data && curl -o data/state_of_the_union.txt https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt
```

### 2. Run the RAG Pipeline

Execute the main script to process the document and answer a query:

```bash
python main.py
```

### 3. View Results

The script will:
1.  Load the document.
2.  Split it into chunks.
3.  Create embeddings and store them in a vector store.
4.  Answer the question: "What did the president say about the economy?"
5.  Print the answer and the trace ID.

## License

MIT
