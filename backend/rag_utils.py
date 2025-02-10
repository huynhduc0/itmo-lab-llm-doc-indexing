import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import requests
from langchain.docstore.document import Document
import pytesseract
from PIL import Image
import pdfplumber
from io import BytesIO
from unstructured.partition.auto import partition
from langchain_core.documents import Document
import docx2txt
from chatbot import generate_toc_with_llm

def load_document(file_path=None, url=None):
    if file_path:
        try:
            # Use partition to get structured elements
            elements = partition(filename=file_path)
            # Check if there are images and handle them (this is just a basic example)

            text = "\n".join([str(el.text) for el in elements])  # Extract text in document order
            
            # OCR - only if needed, and append for now
            ocr_text = extract_text_from_pdf_with_ocr(file_path)
            if ocr_text:
                text += "\nOCR text:\n" + ocr_text
            return elements
           

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    elif url:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        return text
    else:
        raise ValueError("Either file_path or url must be provided.")

def extract_text_from_pdf_with_ocr(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    image = page.to_image().original
                    text += pytesseract.image_to_string(image, lang='eng')
                except Exception as e:
                    print(f"Error OCRing page {i + 1}: {e}")
        return text
    except Exception as e:
        print(f"Error processing PDF with OCR: {e}")
        return None

def create_vector_store(document, doc_name, embedder, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if isinstance(document, list): #Kiểm tra xem document có phải là danh sách các Document hay không
        docs = text_splitter.split_documents([Document(page_content = str(el.text)) for el in document])
    else:
        #Nếu document là string (từ URL), tạo một Document object
        document = [Document(page_content=document)]
        docs = text_splitter.split_documents(document)

    if not docs: # Check if documents are empty
        print("No content in document")
        return None # Return None if the document is empty

    vector_store = FAISS.from_documents(docs, embedder)
    vector_store.save_local(f"document_indexes/{doc_name}")
    return vector_store

def load_vector_store(doc_name, embedder):
    vector_store = FAISS.load_local(f"document_indexes/{doc_name}", embedder, allow_dangerous_deserialization=True)
    return vector_store

def save_uploaded_file(uploaded_file):
    UPLOAD_FOLDER = "uploaded_documents"  # Thư mục lưu trữ tài liệu
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
  
def get_table_of_contents_pdf(file_path):
    try:
        pdf = pdfplumber.open(file_path)
        toc = []
        for page in pdf.pages:
            for element in page.extract_words():
                # Check if the word looks like a heading (e.g., bold, large font)
                if element.get("fontname") and "Bold" in element.get("fontname") and element["size"] > 10:
                    toc.append(element["text"])
        pdf.close()
        return toc
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

def get_table_of_contents_docx(file_path):
    try:
        text = docx2txt.process(file_path)
        return generate_toc_with_llm(text)
    except Exception as e:
        print(f"Error generating TOC with LLM: {e}")
        return None