#test2 - query the chromadb vectorstore
import logging
import os                  #access the chroma folder/files
import re                  #regex to replace multiple spaces
import chromadb            #store data locally for querying

from langchain                  import PromptTemplate
from langchain.embeddings       import HuggingFaceEmbeddings
from langchain.vectorstores     import Chroma
from langchain.chains           import RetrievalQA
from langchain.agents           import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.memory           import ConversationBufferMemory

from langchain.agents           import BaseSingleActionAgent
from langchain.prompts          import StringPromptTemplate
from langchain                  import LLMChain
from typing                     import List, Tuple, Any, Union
from langchain.schema           import AgentAction, AgentFinish

#logging.basicConfig(level=logging.DEBUG)

ABS_PATH    = os.path.dirname(os.path.abspath(__file__))
RUFF_DB_DIR = os.path.join(ABS_PATH, "ruff_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def search(search_input):
    print("search for:", search_input)
    
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
#    print(type(docs)) #list
    content = docs[0][0]
#    print(content) #page_content='Today, Ruff can be used to replace Flake8 when used with any of the following plugins:' metadata={'source': 'https://beta.ruff.rs/docs/faq/'}
    my_str = '"'+str(content)+'"'
#    print(my_str)
    tt = my_str.split("'")
#    print(tt)
#    print(tt[1])
    score = docs[0][1]
#    print(score)   #0.3622860312461853
#    print(docs[0].page_content)
    return tt[1]
#    return docs

def query_chromadb():
    #if the folder does not exist then stop as we cannot do anything without data
    if not os.path.exists(RUFF_DB_DIR):
        raise Exception(f"{RUFF_DB_DIR} does not exist, nothing can be queried")
    
    query = "Why use ruff over flake8?"
    
    tools = [
        Tool(
	    name="Search",
	    func=search,
	    description="useful for when you need to answer questions about RUFF",
        )
    ]
    
    print("Initializing ....")
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = embeddings

    class FakeAgent(BaseSingleActionAgent):
        """Fake Custom Agent."""
    
        @property
        def input_keys(self):
            return ["input"]
    
        def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
        ) -> Union[AgentAction, AgentFinish]:
            """Given input, decided what to do.
    
            Args:
                intermediate_steps: Steps the LLM has taken to date,
                    along with observations
                **kwargs: User inputs.
    
            Returns:
                Action specifying what tool to use.
            """
            return AgentAction(tool="Search", tool_input=kwargs["input"], log="")
    
        async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
        ) -> Union[AgentAction, AgentFinish]:
            """Given input, decided what to do.
    
            Args:
                intermediate_steps: Steps the LLM has taken to date,
                    along with observations
                **kwargs: User inputs.
    
            Returns:
                Action specifying what tool to use.
            """
            return AgentAction(tool="Search", tool_input=kwargs["input"], log="")

    agent = FakeAgent()
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=1)
    
    doc = agent_executor.run(query)
#    print(type(doc)) #str
    #need to convert it into an object so that I can extract the page_content
    print(doc)
    
def main():
    query_chromadb()

if __name__ == '__main__':
    main()
     
