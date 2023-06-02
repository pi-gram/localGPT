# localGPT  (in progress - will be updated when I get around to it)
Ask questions of locally stored documents against an intelligent LLM model that retrieves the data stored within ChromaDB vector databases, using the Langchain framework and AI Agents

Built with [LangChain](https://python.langchain.com/en/latest/), [LlamaCpp](https://github.com/ggerganov/llama.cpp), [Chroma](https://www.trychroma.com/) and [SentenceTransformers](https://www.sbert.net/) although we are using HuggingFaceEmbedding to utilise the [Vicuna model](https://huggingface.co/vicuna/ggml-vicuna-13b-1.1)

# Usage
A sample Python script is provided to explain how to perform the ingestion of data from raw text into a Chroma vector database store.

A sample Python script is provided to show how to perform a simple 1-1 question to get an answer from that stored data and a sample is provided to show how to create a Langchain AI Agent to Execute a Langchain that has access to multiple Chroma vector database stores.

# Output
Outputs will vary, depending upon the model that you use.  Even with the same model, the same question may or may not return the same response.  There is also fun to be had, where you can ask a question and an answer cannot be ascertained from the Chroma data and therefore the model returns a result based upon its internal content.  That is always interesting.


# Disclaimer
THIS IS NOT A SOLUTION FOR YOU TO USE.  It is an experimental exercise for myself that I have built up from figuring out how Langchain works, how to use LLMs locally and how to run questions against local datastores.  I did NOT make this for you to use, it is provided as-is for educational and experimental purposes, therefore do not raise issues expecting me to solve the problems.  If you are stuff use DuckDuckGo search engine (like I have had to do) or the big blobby matter that is contained within your skull to figure it out.  I accept no responsibility for you looking at or using this code.
