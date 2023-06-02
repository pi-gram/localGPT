#test2 - query the 2 chromadbs!
#https://python.langchain.com/en/latest/modules/agents/agents/custom_multi_action_agent.html
import logging
import os                  #access the chroma folder/files
import re                  #regex to replace multiple spaces
import chromadb            #store data locally for querying

from langchain                  import PromptTemplate
from langchain.embeddings       import LlamaCppEmbeddings #pip install llama-cpp-python
#from langchain.embeddings       import HuggingFaceEmbeddings
from langchain.vectorstores     import Chroma
from langchain.chains           import RetrievalQA
from langchain.memory           import ConversationBufferMemory

from langchain.agents           import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.agents           import BaseMultiActionAgent, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts          import StringPromptTemplate
from langchain                  import LLMChain
from typing                     import List, Tuple, Any, Union
from langchain.schema           import AgentAction, AgentFinish

from langchain.tools            import BaseTool


#logging.basicConfig(level=logging.DEBUG)

#tool_names = [tool.name for tool in tools]

ABS_PATH    = os.path.dirname(os.path.abspath(__file__))
RUFF_DB_DIR = os.path.join(ABS_PATH, "ruff_db")

#in order to use and have the model reference working you need to use v0.1.49 and NOT v0.1.50 that is the lateest
#pip3 install llama-cpp-python==0.1.49
#7B model =
#embeddings = HuggingFaceEmbeddings(model_path="/home/tony/dev/tonyGPT/elvis/wip/langchain/ggml-model-q4_0.bin")
embeddings = LlamaCppEmbeddings(model_path="/home/tony/dev/tonyGPT/elvis/wip/langchain/ggml-model-q4_0.bin")
#13B model = 
#embeddings = LlamaCppEmbeddings(model_path="/home/tony/dev/tonyGPT/elvis/wip/langchain/ggml-vicuna-13b-4bit-rev1.bin")

#however, this gets past this error, we get further, we still get same prompt related error below LOL
#
   
#llm_chain = LLMChain(prompt=prompt, llm=llm)
#tool_names = [tool.name for tool in tools]
#
#class CustomOutputParser(AgentOutputParser):
#    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
#        if \"Final Answer:\" in llm_output:
#            print(\"we reached a Final Answer\\n\")
#            return AgentFinish(
#                return_values={\"output\": llm_output.split(\"Final Answer:\")[-1].strip()},
#                log=llm_output,
#            )
#        print(\"We are still looping through potential answers\\n\")
#        # Parse out the action and action input
##        regex = r\"Action\\s*\\d*\\s*:(.*?)\\nAction\\s*\\d*\\s*Input\\s*\\d*\\s*:[\\s]*(.*)\"
##        match = re.search(regex, llm_output, re.DOTALL)
##        print(match)
##        if not match:
##            raise ValueError(f\"Could not parse LLM output: `{llm_output}`\")
##        action = match.group(1).strip()
##        action_input = match.group(2)
#        # Return the action and action input
##        return AgentAction(tool=action, tool_input=action_input.strip(\" \").strip('\"'), log=llm_output)
#        return AgentAction(tool=llm_output.strip(\" \").strip('\"'), #tool_input=llm_output.strip(\" \").strip('\"'), log=llm_output)
#
#    output_parser = CustomOutputParser()


#CHAT_CONVERSATIONAL_REACT_DESCRIPTION = 'chat-conversational-react-description'
#CHAT_ZERO_SHOT_REACT_DESCRIPTION = 'chat-zero-shot-react-description'
#CONVERSATIONAL_REACT_DESCRIPTION = 'conversational-react-description'
#REACT_DOCSTORE = 'react-docstore'
#SELF_ASK_WITH_SEARCH = 'self-ask-with-search'
#ZERO_SHOT_REACT_DESCRIPTION = 'zero-shot-react-description'
#agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
#agent.run("How much did the economy grow by?")
#agent = LLMSingleActionAgent(llm_chain=llm_chain,output_parser=output_parser,stop=["Observation:"])
#agent = LLMSingleActionAgent(llm_chain=llm_chain,
#                             output_parser=output_parser,
#                             stop=["Observation:"],
#                             allowed_tools=tool_names
#                            )
#agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
#agent_executor.run("How much did the economy grow by?")
#

def search_ruff(search_input):
    print("\nwithin search_ruff we are searching for:", search_input)
    
    client_settings = chromadb.config.Settings(
        chroma_db_impl     ="duckdb+parquet",
        persist_directory  =RUFF_DB_DIR,
        anonymized_telemetry=False        #do not allow tracking of data - just incase
    )

    vectorstore = Chroma(
        collection_name    ="ruff_store",
        embedding_function = embeddings,
        client_settings    = client_settings,
        persist_directory  = RUFF_DB_DIR
    )    
    
    docs = vectorstore.similarity_search_with_score(search_input, k=1) #k=1 as we want 1 answer!
    content = docs[0][0]
    my_str = '"'+str(content)+'"'
    tt = my_str.split("'")
    score = docs[0][1]
    return tt[1]

def query_chromadb():
    #if the folder does not exist then stop as we cannot do anything without data
    if not os.path.exists(RUFF_DB_DIR):
        raise Exception(f"{RUFF_DB_DIR} does not exist, nothing can be queried")

        
    query1 = "Why would I use ruff over flake8 and can I purchase it at the IKEA store?"
    
    tools = [
        Tool(
            name = "SearchRuff",
            func=search_ruff,
            description="use for query information about the Python Ruff linter"
        )
    ]
    
    print("Initializing EL.VI.S....")
#    memory = ConversationBufferMemory(memory_key="chat_history")
#keep this here incase we need to do something with history later on

    llm = embeddings #do we need to use the Vicuna model .bin here?


    class FakeAgent(BaseMultiActionAgent):
        """Fake Custom Agent."""
    
        @property
        def input_keys(self):
            return ["input"]
    
        def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
        ) -> Union[AgentAction, AgentFinish]:
            """Given input, decided what to do. If you cannot provide a correct answer do not guess.
    
            Args:
                intermediate_steps: Steps the LLM has taken to date,
                    along with observations
                **kwargs: User inputs.
    
            Returns:
                Action specifying what tool to use.
            """
            if len(intermediate_steps) == 0:
                return [
                    AgentAction(tool="SearchRuff", tool_input=kwargs["input"], log=""),
                ]
            else:
                return AgentFinish(return_values={"output": "\nplan - I have absolutely no idea"}, log="")
    

        async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
        ) -> Union[List[AgentAction], AgentFinish]:
            """Given input, decided what to do. If you cannot provide a correct answer do not guess.

            Args:
                intermediate_steps: Steps the LLM has taken to date,
                    along with observations
                **kwargs: User inputs.

            Returns:
                Action specifying what tool to use.
            """
            if len(intermediate_steps) == 0:
                return [
                    AgentAction(tool="SearchRuff", tool_input=kwargs["input"], log=""),
                ]
            else:
                return AgentFinish(return_values={"output": "\naplan - I have no idea"}, log="")

   
    agent = FakeAgent()
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=1)

#we can call this directly and the output will be shown in the command-line:
#    agent_executor.run(query1)
#or we can contain the responses into a single string and do something with the result
    doc = agent_executor.run(query1)
#    print(type(doc)) #str
    print(doc)


 
   
def main():
    query_chromadb()

if __name__ == '__main__':
    main()
     
