# localGPT  (in progress - will be updated when I get around to it)
Ask questions of locally stored documents against an intelligent LLM model that retrieves the data stored within ChromaDB vector databases, using the Langchain framework and AI Agents

Built with [LangChain](https://python.langchain.com/en/latest/), [LlamaCpp](https://github.com/ggerganov/llama.cpp), [Chroma](https://www.trychroma.com/) and [SentenceTransformers](https://www.sbert.net/) although we are using HuggingFaceEmbedding to utilise the [Vicuna model](https://huggingface.co/vicuna/ggml-vicuna-13b-1.1)

# Environment
There's nothing too special, I initially set this up and ran it on Windows 10, before that spectacularly crashed the PC and no longer would boot, therefore I switched over to UBuntu Linux 22.04, of which I have had no problems.  I could list, and I might do at some point, what all the dependencies and steps you need to set this up, such as installing langchain, llamacpp, etc..etc... but I would hope that if you are reading this, you already know, if not, you may be disappointed that I am not teaching you - go find a Udemy course or a teacher / mentor somewhere that can help you, that person is not me.  Bit harsh: so I might update this later.
I give reference within the Python code to the vicuna model itself, but I have not uploaded it, as it is 8GB, you can source it from the link above and download it and place it in your location - or even change the model entirely.

# Usage
A sample Python 3 script is provided to explain how to perform the ingestion of data from raw text into a Chroma vector database store.

A sample Python 3 script is provided to show how to perform a simple 1-1 question to get an answer from that stored data and a sample is provided to show how to create a Langchain AI Agent to Execute a Langchain that has access to multiple Chroma vector database stores.

reuters_news.txt - file contains a snpipet of 10 news articles extracted into a format that can be ingested.  This could be any text from anywhere, so long as it has text within it, you can ingest it - easy for you to extend.

reuters_news_snippet_ingest.py - reads the above reuters_news.txt file, uses the grammar from the vicuna model to store within a vector database, it creates the folder to place the files within and generates the tokens from the text in the file.  You'll note, I did not make anything set from environment variable settings, it's all hard-coded, deal with it.

reuters_news_snippet_query.py - takes a question and asks it of the newly created Chroma vector database store - that code is horrid in the way it chops up the response (just looks for ' chars), so that can be improved, but it demonstrates that it IS querying the stored data and not using the model to answer the question.  There's a huge mish-mash of sample code lifted from articles and Langchain examples merged together to get this to work.

testx.py files are just that, just experimental files that show the evolution of trying things out to get to where we need to be.


# Output
Outputs will vary, depending upon the model that you use.  Even with the same model, the same question may or may not return the same response.  There is also fun to be had, where you can ask a question and an answer cannot be ascertained from the Chroma data and therefore the model returns a result based upon its internal content.  That is always interesting.  I also thought I had the simple Langchain samples where you just perform .query() against the Agent and you get a zero_shot answer, maybe I deleted those test files? anyway, might add them back in; they were not what I was looking to achieve, which is why I moved onto the .Chain() function as I'm really looking at how do I access multipe stores and can the Langchain use the [tools] to query them all to get the answer.


# Disclaimer
THIS IS NOT A SOLUTION FOR YOU TO USE.  It is an experimental exercise for myself that I have built up from figuring out how Langchain works, how to use LLMs locally and how to run questions against local datastores.  I did NOT make this for you to use, it is provided as-is for educational and experimental purposes, therefore do not raise issues expecting me to solve the problems.  If you are stuck, use DuckDuckGo search engine (like I have had to do) or the big blobby matter that is contained within your skull to figure it out.  I accept no responsibility for you looking at or using this code.
