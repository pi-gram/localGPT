#test1 - ingest data into chromadb
import json                #handle JSON data
import logging
import os                  #access the chroma folder/files
import re                  #regex to replace multiple spaces
import chromadb            #store data locally for querying

from langchain.embeddings       import LlamaCppEmbeddings
from langchain.text_splitter    import RecursiveCharacterTextSplitter
from langchain.vectorstores     import Chroma
from langchain.document_loaders import WebBaseLoader

logging.basicConfig(level=logging.DEBUG)

ABS_PATH    = os.path.dirname(os.path.abspath(__file__))
VIC_RUFF_DB_DIR = os.path.join(ABS_PATH, "vic_ruff_db")

def replace_newlines_and_spaces(text):
    #replace all newline characters with spaces
    text = text.replace("\n", " ")
    #replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text

def init_chromadb():
    #if the folder does not exist then make the folder to store the output into
    if not os.path.exists(VIC_RUFF_DB_DIR):
        os.mkdir(VIC_RUFF_DB_DIR)
    client_settings = chromadb.config.Settings(
        chroma_db_impl     ="duckdb+parquet",
        persist_directory  =VIC_RUFF_DB_DIR,
        anonymized_telemetry=False        #do not allow tracking of data - just incase
    )
 
    embeddings = LlamaCppEmbeddings(model_path="/home/tony/dev/tonyGPT/elvis/wip/langchain/ggml-vicuna-13b-4bit-rev1.bin")
 
    vectorstore = Chroma(
        collection_name    ="vic_ruff_store",
        embedding_function = embeddings,
        client_settings    = client_settings,
        persist_directory  = VIC_RUFF_DB_DIR
    )
    loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")
    docs   = loader.load()
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    vic_ruff_texts    = text_splitter.split_documents(docs)
    
    print("about to add text to Chroma VIC_RUFF DB")
    vectorstore.add_documents(documents=vic_ruff_texts, embedding=embeddings)
    vectorstore.persist()
    print(vectorstore)
    
def main():
    init_chromadb()

if __name__ == '__main__':
    main()
     
