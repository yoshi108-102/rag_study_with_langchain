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



