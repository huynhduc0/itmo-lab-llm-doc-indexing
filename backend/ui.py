import dotenv
dotenv.load_dotenv()
import streamlit as st
import os
import tempfile
from rag_utils import load_document, create_vector_store, load_vector_store
from chatbot import init_chatbot
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from bs4 import BeautifulSoup
import requests
import pdfplumber

# Khởi tạo Embeddings (hoặc GeminiEmbeddings nếu bạn muốn)
EMBEDDER = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
UPLOAD_FOLDER = "uploaded_documents"  # Thư mục lưu trữ tài liệu

def get_available_docs():
    return [f for f in os.listdir('document_indexes') if not f.startswith('.')]

def save_uploaded_file(uploaded_file):
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
def get_table_of_contents(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all heading tags (h1, h2, h3, etc.)
        heading_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        toc = []
        for tag in heading_tags:
            toc.append({"text": tag.text, "id": tag.get('id')})
        return toc
    except:
        return None

def get_table_of_contents_pdf(file_path):
    try:
        pdf = pdfplumber.open(file_path)
        toc = []
        for page in pdf.pages:
            for element in page.extract_words():
                # Check if the word looks like a heading (e.g., bold, large font)
                if element["fontname"] and "Bold" in element["fontname"] and element["size"] > 10:
                    toc.append(element["text"])
        pdf.close()
        return toc
    except:
        return None
def main():
    # Khởi tạo các biến trong session state
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}  # Lưu trữ thông tin phiên
    if 'current_session' not in st.session_state:
        st.session_state.current_session = None # Khởi tạo là None
    if 'all_docs_messages' not in st.session_state:
        st.session_state.all_docs_messages = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'doc_name' not in st.session_state:
        st.session_state.doc_name = None
    if 'chat_mode' not in st.session_state:
        st.session_state.chat_mode = "File hiện tại"  # Mặc định chat trên file hiện tại

    st.title("Docs Assistant")

    with st.sidebar:
        st.title('DocAssistant')

        st.markdown('## New Document')
        file = st.file_uploader(label='')
        applied = st.button(label='📄 Load Document')  # Button Load Document

        # if apply button clicked
        if applied:
            # if no file uploaded, warn the user
            if not file:
                st.warning('Please load a document.')
            # if a file is uplaoded, read and create the embedding
            else:
                with st.spinner():
                    # Lưu file tải lên
                    file_path = save_uploaded_file(file)
                    doc_name = file.name[:file.name.index('.')]
                    st.session_state.doc_name = doc_name

                    # Tạo thư mục indexes nếu chưa có
                    if not os.path.exists('document_indexes'):
                        os.makedirs('document_indexes')

                    if not os.path.exists(doc_name):
                        os.makedirs(doc_name)
                    if doc_name.strip() in get_available_docs():
                        st.warning('The document is already exists.')
                        st.session_state.vectorstore = load_vector_store(doc_name, EMBEDDER)
                    else:
                        doc = load_document(file_path=file_path)
                        st.session_state.vectorstore = create_vector_store(doc, doc_name, EMBEDDER)
                        st.success('Document load successfully ✔️')
                # Xóa file tạm sau khi xử lý xong
                # os.remove(file_path) # <--  Bạn có thể bỏ dòng này nếu muốn giữ lại file tạm

        st.markdown('## Chat Sessions')
        with st.form("new_session_form", clear_on_submit=True):
            session_name = st.text_input("Session Name", "Session")
            selected_docs = st.multiselect("Select Documents", get_available_docs(), key="selected_docs")
            create_session = st.form_submit_button("Create New Session")

            if create_session:
                session_key = session_name.replace(" ", "_") # Tạo key từ tên session
                if session_key in st.session_state.chat_sessions:
                    st.warning(f"Session '{session_name}' already exists. Please use a different name.")
                else:
                    st.session_state.chat_sessions[session_key] = selected_docs
                    st.session_state.current_session = session_key
                    # Cập nhật vector store khi tạo session mới
                    if selected_docs:
                        # Load vector store của file đầu tiên trong danh sách
                        st.session_state.vectorstore = load_vector_store(selected_docs[0], EMBEDDER)
                    else:
                        st.session_state.vectorstore = None # Set về None nếu không có file nào được chọn

        session_options = list(st.session_state.chat_sessions.keys())
        if session_options:
            new_session = st.selectbox("Select Session", session_options, key="session_selectbox")
            if new_session != st.session_state.current_session:
                st.session_state.current_session = new_session
                selected_docs = st.session_state.chat_sessions[new_session]
                # Cập nhật vector store khi chọn session
                if selected_docs:
                    st.session_state.vectorstore = load_vector_store(selected_docs[0], EMBEDDER)
                else:
                    st.session_state.vectorstore = None

        else:
            st.info("Create a chat session to start.")

    if st.session_state.current_session: # Đảm bảo đã chọn session rồi
        with st.expander(f"Session Information: {st.session_state.current_session}"):
            #Show selected documents
            selected_docs = st.session_state.chat_sessions.get(st.session_state.current_session, [])
            if selected_docs:
                st.markdown("**Selected Documents:**")
                for doc in selected_docs:
                    st.markdown(f"- {doc}")
                    if "http" in doc:
                        toc = get_table_of_contents(doc)
                    else:
                        toc = get_table_of_contents_pdf(doc)
                    if toc:
                         st.markdown("**Table of Contents**")
                         for item in toc:
                            st.markdown(f"- {item}")

            else:
                st.info("No document selected in this session")

            new_file = st.file_uploader("Add more documents", key="add_file_to_session")
            if new_file is not None:
                with st.spinner():
                    file_path = save_uploaded_file(new_file)
                    new_doc_name = new_file.name[:new_file.name.index('.')]
                    doc = load_document(file_path=file_path)
                    new_vectorstore = create_vector_store(doc, new_doc_name, EMBEDDER)

                    #Cập nhật lại session state
                    st.session_state.chat_sessions[st.session_state.current_session].append(new_doc_name)
                    #Cập nhật vector store nếu cần
                    st.session_state.vectorstore = load_vector_store(new_doc_name, EMBEDDER)
                    st.success(f'Document {new_doc_name} load successfully ✔️')

        messages_key = f"{st.session_state.current_session}_messages"

        if messages_key not in st.session_state:
            st.session_state[messages_key] = []

        for message in st.session_state[messages_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        

        if st.session_state.vectorstore is not None:
            qa_chain = init_chatbot(st.session_state.vectorstore)

            if query := st.chat_input("Ask a question about the document"):  # Kiểm tra lại chỗ này
                st.session_state[messages_key].append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)
                answer = qa_chain.invoke({"input": query})
                if hasattr(answer, 'content'):
                    st.chat_message("assistant").markdown(answer.content)
                    st.session_state[messages_key].append({"role": "assistant", "content": answer.content})
                else:
                    st.chat_message("assistant").markdown(answer)
                    st.session_state[messages_key].append({"role": "assistant", "content": answer})

        else:
            st.info("Please load a document")