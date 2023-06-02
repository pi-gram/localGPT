#test1 - ingest data into chromadb
import json                #handle JSON data
import logging
import os                  #access the chroma folder/files
import re                  #regex to replace multiple spaces
import chromadb            #store data locally for querying

from fastapi.encoders           import jsonable_encoder
#from langchain.document_loaders import PyPDFLoader # pip install pypdf
from langchain.embeddings       import HuggingFaceEmbeddings
from langchain.text_splitter    import RecursiveCharacterTextSplitter
from langchain.vectorstores     import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain.chains           import RetrievalQA

logging.basicConfig(level=logging.DEBUG)

ABS_PATH    = os.path.dirname(os.path.abspath(__file__))
IKEA_DB_DIR = os.path.join(ABS_PATH, "ikea_db")

def replace_newlines_and_spaces(text):
    #replace all newline characters with spaces
    text = text.replace("\n", " ")
    #replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text

def init_chromadb():
    #if the folder does not exist then make the folder to store the output into
    if not os.path.exists(IKEA_DB_DIR):
        os.mkdir(IKEA_DB_DIR)
    client_settings = chromadb.config.Settings(
        chroma_db_impl     ="duckdb+parquet",
        persist_directory  =IKEA_DB_DIR,
        anonymized_telemetry=False        #do not allow tracking of data - just incase
    )
#    model_bin  = "/home/tony/dev/tonyGPT/gpt4all/model/ggml-vicuna-13b-1.1-q4_2.bin"
#    embeddings = HuggingFaceEmbeddings(model_path=model_bin)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        collection_name    ="ikea_store",
        embedding_function = embeddings,
        client_settings    = client_settings,
        persist_directory  = IKEA_DB_DIR
    )
    loader = WebBaseLoader("https://www.ikea.com/us/en/customer-service/faq/")
    docs   = loader.load()
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    ikea_texts    = text_splitter.split_documents(docs)
    
    print("about to add text to Chroma IKEA DB")
    vectorstore.add_documents(documents=ikea_texts, embedding=embeddings)
    vectorstore.persist()
    print(vectorstore)
    
def main():
    init_chromadb()

if __name__ == '__main__':
    main()
     
