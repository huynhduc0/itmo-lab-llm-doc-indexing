from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import requests
import os

def load_document(file_path=None, url=None):
    if file_path:
      loader = UnstructuredFileLoader(file_path)
      document = loader.load()
    elif url:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        return text
    else:
        raise ValueError("Either file_path or url must be provided.")
    return document

def create_vector_store(document, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if isinstance(document, list): # document is from file loader
        docs = text_splitter.split_documents(document)
    else: # document is from web loader
        docs = text_splitter.split_text(document)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
    if isinstance(docs, list):
      vector_store = FAISS.from_documents(docs, embeddings)
    else:
        vector_store = FAISS.from_texts(docs, embeddings)
    return vector_store