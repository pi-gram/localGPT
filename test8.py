#test8 - Langchain, with customtemplate, customparser and ready to now use ChromaDB
#https://python.langchain.com/en/latest/modules/agents/agents/custom_multi_action_agent.html
import logging
import os                  #access the chroma folder/files
import re                  #regex to replace multiple spaces
import chromadb            #store data locally for querying

from langchain                  import PromptTemplate, LLMChain
from langchain                  import LlamaCpp
from langchain.vectorstores     import Chroma
from langchain.embeddings       import LlamaCppEmbeddings

from langchain.agents           import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts          import StringPromptTemplate
from langchain.schema           import AgentAction, AgentFinish
from langchain.tools            import BaseTool
from typing                     import List, Union

ABS_PATH    = os.path.dirname(os.path.abspath(__file__))
RUFF_DB_DIR = os.path.join(ABS_PATH, "ruff_db")
IKEA_DB_DIR = os.path.join(ABS_PATH, "ikea_db")

query1 = "Why would I use ruff instead of flake8?"

#use different llm for using ruff embedding_function?
llm_embed = LlamaCppEmbeddings(model_path="/home/tony/dev/tonyGPT/elvis/wip/langchain/ggml-vicuna-13b-4bit-rev1.bin")
    
def search_ruff(input):
    print("\nWithin search_ruff we are searching for:", input)
    
    client_settings = chromadb.config.Settings(
        chroma_db_impl     ="duckdb+parquet",
        persist_directory  =RUFF_DB_DIR,
        anonymized_telemetry=False        #do not allow tracking of data - just incase
    )

    vectorstore = Chroma(
        collection_name    ="ruff_store",
        embedding_function = llm_embed, #llm,
        client_settings    = client_settings,
        persist_directory  = RUFF_DB_DIR
    )    
    
    print("\nabout to execute vectorstore\n")
    docs = vectorstore.similarity_search_with_score(input, k=1) #k=1 as we want 1 answer!
#    return docs
    content = docs[0][0]
    my_str = '"'+str(content)+'"'
    tt = my_str.split("'")
#    score = docs[0][1]
    print(tt[1])
    return tt[1]

tools = [
#    Tool(
#        name = "SearchIKEA",
#        func=search_ikea,
#        description="use for query information about the IKEA FAQ"
#    ),
    Tool(
        name = "SearchRuff",
        func=search_ruff,
        description="use for query information about the Python Ruff linter"
    ),
#    Tool(
#        name = "SearchJanes",
#        func=search_janes,
#        description="call this to query the JANES database"
#    ),
]
tool_names = [tool.name for tool in tools]


# Set up the base template
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question


Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)



#set the llm
#7B model =
#llm = LlamaCpp(model_path="/home/tony/dev/tonyGPT/elvis/wip/langchain/ggml-model-q4_0.bin")
#13B model = 
llm = LlamaCpp(model_path="/home/tony/dev/tonyGPT/elvis/wip/langchain/ggml-vicuna-13b-4bit-rev1.bin")

#let's use the sentenceTransformer model instead of Vicuna and see what happens
#llm = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#
#Can't instantiate abstract class BaseLanguageModel with abstract methods agenerate_prompt, generate_prompt (type=type_error)
#
#ah ha! so this returns us all the way back to the paramter issue - so THIS is the real reason why we couldn't "cross the streams" earlier...
#it kind of makes sense, so let's stick with the LlamaCpp(model) even if it is more resource intensive.  Just need to figure out how to get the code to use the ruff_store to answer the question rather than the Vicuna internal model - thaat's the challenge....

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


output_parser = CustomOutputParser()

    
print("Initializing EL.VI.S....")

llm_chain = LLMChain(prompt=prompt, llm=llm)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,     #need to define the output_parser
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

print(f"QUESTION: {query1}")
agent_executor.run(query1)
#result = agent_executor.run(query1)
##add a try catch here to catch if it somehow decides to throw an error?
#print(result)

