チャンキングしてみた

chunks[0]の中身
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,add_start_index=True)
chunks = text_splitter.split_documents(docs)
print(chunks[0])
```
`Recursive`ってなんやねんと思ったが、
https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter

によると、`RecursiveTextSplitter`とかいうのは、文字の段落とかの文章的な区切りを大事にしながら区切っていくためのものらしい。
具体的な実装はよくみていないが、

The default list is ["\n\n", "\n", " ", ""].

とか書いてあるので、多分リストの上から順にdfs的に区切っていく感じなんだろう。


Date: June 23, 2023  |  Estimated Reading Time: 31 min  |  Author: Lilian Weng


Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.
Agent System Overview#
In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:

Planning

Subgoal and decomposition: The agent breaks down large tasks into smaller, manageable subgoals, enabling efficient handling of complex tasks.
Reflection and refinement: The agent can do self-criticism and self-reflection over past actions, learn from mistakes and refine them for future steps, thereby improving the quality of final results.


Memory' metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'start_index': 8}
```

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
```

https://reference.langchain.com/python/langchain/tools/?_gl=1*t8ik2c*_gcl_au*MTc0MjQ4NDU0OC4xNzcxNDk5MDYz*_ga*MTIwNjUxNjAyMi4xNzcxNDk5MDYz*_ga_47WX3HKKY2*czE3NzE0OTkwNjMkbzEkZzEkdDE3NzE1MDI3MTUkajYwJGwwJGgw#langchain.tools.tool


toolデコレータを使用すると、どうやらfunction callingを自分でカスタマイズできる？

https://zenn.dev/pharmax/articles/1b351b730eef61

どうもvector_storeの検索は、今回は類似度検索とかいうのを使用しているので、多分コサイン類似度なんだろうな。
そういえば、コサイン類似度の生成の際に、ナイーブな手法だとword2vecとかを使用していた気がするが、今回は何を使用しているのだろうか。

```bash
================================ Human Message =================================

What is task decomposition?
================================== Ai Message ==================================
Tool Calls:
  retrieve_context (call_rXJ6lFfRRCx5PJzzkgonS0DB)
 Call ID: call_rXJ6lFfRRCx5PJzzkgonS0DB
  Args:
    query: task decomposition
================================= Tool Message =================================
Name: retrieve_context

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'start_index': 2578}
Content: Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.
Another quite distinct approach, LLM+P (Liu et al. 2023), involves relying on an external classical planner to do long-horizon planning. This approach utilizes the Planning Domain Definition Language (PDDL) as an intermediate interface to describe the planning problem. In this process, LLM (1) translates the problem into “Problem PDDL”, then (2) requests a classical planner to generate a PDDL plan based on an existing “Domain PDDL”, and finally (3) translates the PDDL plan back into natural language. Essentially, the planning step is outsourced to an external tool, assuming the availability of domain-specific PDDL and a suitable planner which is common in certain robotic setups but not in many other domains.
Self-Reflection#

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'start_index': 1638}
Content: Component One: Planning#
A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.
Task Decomposition#
Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.
Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.
================================== Ai Message ==================================

Task decomposition is a process where a complicated task is broken down into smaller and simpler steps. This technique is essential for agents to understand the steps involved in a task and plan ahead. One method of task decomposition is the Chain of Thought (CoT), which prompts models to "think step by step" to decompose complex tasks into more manageable ones. Additionally, the Tree of Thoughts extends CoT by exploring multiple reasoning possibilities at each step, creating a tree structure of thought steps and multiple thoughts per step. This process aids in interpreting the model's thinking process and enhances performance on complex tasks.
```

普通にopenaiのエンコーダに通せばベクトライズされるんじゃね？？？多分そう。

toolデコレータはpydocsとかを取得したりして面白いのでコードを追うといいらしい

