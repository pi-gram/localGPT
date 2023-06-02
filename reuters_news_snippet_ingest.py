#test1 - ingest data into chromadb
import json                #handle JSON data
import logging
import os                  #access the chroma folder/files
import re                  #regex to replace multiple spaces
import chromadb            #store data locally for querying

from langchain.embeddings       import LlamaCppEmbeddings
from langchain.text_splitter    import RecursiveCharacterTextSplitter
from langchain.vectorstores     import Chroma
#from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import TextLoader

logging.basicConfig(level=logging.DEBUG)

ABS_PATH    = os.path.dirname(os.path.abspath(__file__))
REUTERS_NEWS_DB_DIR = os.path.join(ABS_PATH, "reuters_news_db")

def replace_newlines_and_spaces(text):
    #replace all newline characters with spaces
    text = text.replace("\n", " ")
    #replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text

def init_chromadb():
    #if the folder does not exist then make the folder to store the output into
    if not os.path.exists(REUTERS_NEWS_DB_DIR):
        os.mkdir(REUTERS_NEWS_DB_DIR)
    client_settings = chromadb.config.Settings(
        chroma_db_impl     ="duckdb+parquet",
        persist_directory  =REUTERS_NEWS_DB_DIR,
        anonymized_telemetry=False        #do not allow tracking of data - just incase
    )
 
    embeddings = LlamaCppEmbeddings(model_path="/home/tony/dev/tonyGPT/elvis/wip/langchain/ggml-vicuna-13b-4bit-rev1.bin")
 
    vectorstore = Chroma(
        collection_name    ="reuters_news_store",
        embedding_function = embeddings,
        client_settings    = client_settings,
        persist_directory  = REUTERS_NEWS_DB_DIR
    )
    
    loader = TextLoader('reuters_news.txt', encoding='utf8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    reuters_news_texts = text_splitter.split_documents(documents)

#    with open("reuters_news.txt") as f:
#        reuters_news = f.read()
#    docs = replace_newlines_and_spaces(reuters_news)
#    print(docs)
    
#    loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")
#    docs   = loader.load()
#    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#    reuters_news_texts    = text_splitter.split_documents(docs)
    
    print("about to add text to Chroma REUTERS_NEWS_DB_DIR")
    vectorstore.add_documents(documents=reuters_news_texts, embedding=embeddings)
    vectorstore.persist()
    print(vectorstore)
    
def main():
    init_chromadb()

if __name__ == '__main__':
    main()
     
