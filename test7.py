#test7 - Langchain but now with ChromaDB

from langchain  import LlamaCpp
from langchain  import PromptTemplate, LLMChain


#setup prompt info
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

#set the llm
#7B model =
llm = LlamaCpp(model_path="/home/tony/dev/tonyGPT/elvis/wip/langchain/ggml-model-q4_0.bin")
#13B model = 
#llm = LlamaCpp(model_path="/home/tony/dev/tonyGPT/elvis/wip/langchain/ggml-vicuna-13b-4bit-rev1.bin")

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
print(f"QUESTION = {question}\n")

result = llm_chain.run(question)

print(f"\nRESULT = {result}")

