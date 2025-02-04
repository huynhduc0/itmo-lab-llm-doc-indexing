import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import requests
from langchain.docstore.document import Document
def load_document(file_path=None, url=None):
    if file_path:
        loader = UnstructuredFileLoader(file_path)
        document = loader.load()
        return document
    elif url:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        return text
    else:
        raise ValueError("Either file_path or url must be provided.")

def create_vector_store(document, doc_name, embedder, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if isinstance(document, list): #Kiểm tra xem document có phải là danh sách các Document hay không
        docs = text_splitter.split_documents(document)
    else:
        #Nếu document là string (từ URL), tạo một Document object
        document = [Document(page_content=document)]
        docs = text_splitter.split_documents(document)

    vector_store = FAISS.from_documents(docs, embedder)
    vector_store.save_local(f"document_indexes/{doc_name}")
    return vector_store

def load_vector_store(doc_name, embedder):
    vector_store = FAISS.load_local(f"document_indexes/{doc_name}", embedder, allow_dangerous_deserialization=True)
    return vector_store