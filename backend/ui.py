import streamlit as st
import os
from rag_utils import load_document, create_vector_store, load_vector_store, save_uploaded_file
from chatbot import init_chatbot, generate_toc_with_llm
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from bs4 import BeautifulSoup
import requests
import pdfplumber
import docx2txt
import dotenv
from PIL import Image
from database import SessionLocal, ChatSession, ChatMessage, DocumentInfo, DocumentTOC
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from typing import List, Dict
import markdown

dotenv.load_dotenv()

EMBEDDER = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
UPLOAD_FOLDER = "uploaded_documents"
CHUNK_FOLDER = "document_chunks"

engine = create_engine(os.getenv("DB_URI", "sqlite:///./doc_assistant.db"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_available_docs():
    return [f for f in os.listdir('document_indexes') if not f.startswith('.') and not os.path.isfile(os.path.join('document_indexes', f))]


def get_table_of_contents(url):
    try:
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        webpage_content = soup.get_text()
        return generate_toc_with_llm(webpage_content)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return []


def get_table_of_contents_pdf(file_path):
    try:
        doc_content = load_document(file_path=file_path)
        return generate_toc_with_llm(doc_content)
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


def chunk_document_by_toc(document_content: str, toc: List[str], doc_name: str) -> Dict[str, str]:
    """
    Chunks the document content, saves each chunk to a separate MD file, preserving Markdown format,
    and returns a dictionary where the keys are TOC titles and the values are the filenames of the
    corresponding MD files.
    """
    chunks = {}
    if not os.path.exists(CHUNK_FOLDER):
        os.makedirs(CHUNK_FOLDER)

    for i, item in enumerate(toc):
        title = item.replace("/", "_").replace("\\", "_").replace(":", "_")
        start_index = document_content.find(item)
        if start_index != -1:
            if i < len(toc) - 1:
                next_title = toc[i + 1]
                end_index = document_content.find(next_title, start_index)
                if end_index != -1:
                    chunk_content = document_content[start_index:end_index]
                else:
                    chunk_content = document_content[start_index:]
            else:
                chunk_content = document_content[start_index:]

            filename = f"{doc_name}_{title}.md"
            filepath = os.path.join(CHUNK_FOLDER, filename)
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(chunk_content)  # Save with original Markdown formatting
                chunks[item] = filename
            except Exception as e:
                print(f"Error saving chunk to file: {e}")
                chunks[item] = None

    return chunks


def load_document_md(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading markdown file: {e}")
        return None


def process_document(db, file=None, url=None):
    if not file and not url:
        st.warning('Please load a document or enter a URL.')
        return None

    with st.spinner():
        if file:
            file_path = save_uploaded_file(file)
            doc_name = file.name[:file.name.index('.')]
            st.session_state.doc_name = doc_name
            st.session_state.doc_url = None

            try:
                file_extension = file.name.split('.')[-1].lower()

                if file_extension == 'md':
                    doc_content = load_document_md(file_path)
                else:
                    doc_content = load_document(file_path=file_path)

                if isinstance(doc_content, list):
                    doc_content_str = "\n".join([str(item) for item in doc_content])
                else:
                    doc_content_str = str(doc_content)

                toc = generate_toc_with_llm(doc_content_str)
                st.session_state.doc_toc = toc

                st.session_state.doc_chunks = chunk_document_by_toc(doc_content_str, toc, doc_name)

                elements = load_document(file_path=file_path)
                vectorstore = create_vector_store(elements, doc_name, EMBEDDER)

                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.success('Vector store successfully ✔️')

                if toc:
                    existing_toc = db.query(DocumentTOC).filter(DocumentTOC.doc_name == doc_name).first()
                    if existing_toc:
                        existing_toc.toc_items = toc
                    else:
                        db_toc = DocumentTOC(doc_name=doc_name, toc_items=toc)
                        db.add(db_toc)
                    db.commit()
                    st.session_state.doc_toc = toc
                else:
                    st.session_state.doc_toc = None

                return

            except Exception as e:
                db.rollback()
                st.error(f"Error loading {file_path}: {e}")
                return None

        elif url:
            st.session_state.doc_url = url
            doc = load_document(url=url)
            doc_name = url.replace("/", "_").replace(":", "_")
            st.session_state.doc_name = doc_name

            if isinstance(doc, list):
                doc_content_str = "\n".join([str(item) for item in doc])
            else:
                doc_content_str = str(doc_content)

            toc = get_table_of_contents(url)
            st.session_state.doc_chunks = chunk_document_by_toc(doc_content_str, toc, doc_name)

            if toc:
                existing_toc = db.query(DocumentTOC).filter(DocumentTOC.doc_name == doc_name).first()
                if existing_toc:
                    existing_toc.toc_items = toc
                else:
                    db_toc = DocumentTOC(doc_name=doc_name, toc_items=toc)
                    db.add(db_toc)
                db.commit()
                st.session_state.doc_toc = toc
            else:
                st.session_state.doc_toc = None

            st.session_state.vectorstore = create_vector_store(doc, doc_name, EMBEDDER)
        st.success('Document load successfully ✔️')
        return st.session_state.vectorstore


def display_messages(messages_key):
    if messages_key in st.session_state:
        for message in st.session_state[messages_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def get_session_options(db):
    sessions = db.query(ChatSession).all()
    return {session.session_key: session for session in sessions}


def delete_session(db, session_key):
    try:
        session_to_delete = db.query(ChatSession).filter(ChatSession.session_key == session_key).first()

        if session_to_delete:
            db.query(ChatMessage).filter(ChatMessage.session_key == session_key).delete()
            db.delete(session_to_delete)
            db.commit()
            st.success(f"Session '{session_key}' and its messages deleted successfully!")
            return True
        else:
            st.warning(f"Session '{session_key}' not found.")
            return False
    except Exception as e:
        db.rollback()
        st.error(f"Error deleting session '{session_key}': {e}")
        return False


def main():
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}
    if 'current_session' not in st.session_state:
        st.session_state.current_session = None
    if 'all_docs_messages' not in st.session_state:
        st.session_state.all_docs_messages = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'doc_name' not in st.session_state:
        st.session_state.doc_name = None
    if 'doc_url' not in st.session_state:
        st.session_state.doc_url = None
    if 'doc_toc' not in st.session_state:
        st.session_state.doc_toc = None
    if 'doc_chunks' not in st.session_state:
        st.session_state.doc_chunks = {}
    if 'chat_mode' not in st.session_state:
        st.session_state.chat_mode = "File hiện tại"
    if 'deep_reasoning' not in st.session_state:
        st.session_state.deep_reasoning = False
    if 'internet_search' not in st.session_state:
        st.session_state.internet_search = False
    if 'elements' not in st.session_state:
        st.session_state.elements = None
    if 'modal_open' not in st.session_state:
        st.session_state.modal_open = False
    if 'selected_toc_item' not in st.session_state:
        st.session_state.selected_toc_item = None
    if 'show_doc_section' not in st.session_state:
        st.session_state.show_doc_section = False
    if 'show_db_admin' not in st.session_state:
        st.session_state.show_db_admin = False
    if 'delete_session_key' not in st.session_state:
        st.session_state.delete_session_key = None
    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = False

    st.title("Docs Assistant")

    with st.sidebar:
        st.title('DocAssistant')

        st.markdown('## New Document')
        file = st.file_uploader(label='')
        url = st.text_input("Enter a URL", key="url_input")
        applied = st.button(label='📄 Load Document')

        if applied:
            with SessionLocal() as db:
                try:
                    st.session_state.vectorstore = process_document(db=db, file=file, url=url)
                except Exception as e:
                    db.rollback()
                    st.error(f"Error during document loading: {e}")
                finally:
                    db.close()

        st.markdown('## Chat Sessions')
        with st.form("new_session_form", clear_on_submit=True):
            session_name = st.text_input("Session Name", "Session")
            selected_docs = st.multiselect("Select Documents", get_available_docs(), key="selected_docs")
            create_session = st.form_submit_button("Create New Session")

            if create_session:
                with SessionLocal() as db:
                    session_key = session_name.replace(" ", "_")
                    if session_key in st.session_state.chat_sessions:
                        st.warning(f"Session '{session_name}' already exists. Please use a different name.")
                    else:
                        st.session_state.chat_sessions[session_key] = selected_docs
                        db_session = ChatSession(session_key=session_key, selected_docs=selected_docs)
                        db.add(db_session)
                        db.commit()
                        st.session_state.current_session = session_key

                        if selected_docs:
                            st.session_state.vectorstore = load_vector_store(selected_docs[0], EMBEDDER)
                            db_toc = db.query(DocumentTOC).filter(DocumentTOC.doc_name == selected_docs[0]).first()
                            if db_toc:
                                st.session_state.doc_toc = db_toc.toc_items
                            else:
                                st.session_state.doc_toc = None
                        else:
                            st.session_state.vectorstore = None
                            st.session_state.doc_toc = None

        with SessionLocal() as db:
            session_options = get_session_options(db)
            session_keys = list(session_options.keys())

            with st.expander("Session List", expanded=True):
                for session_key in session_keys:
                    cols = st.columns([0.7, 0.3])
                    with cols[0]:
                        if st.button(session_key, key=f"session_{session_key}"):
                            st.session_state.current_session = session_key
                    with cols[1]:
                        if st.button("Delete", key=f"delete_{session_key}"):
                            st.session_state.delete_session_key = session_key
                            st.session_state.confirm_delete = True

            if st.session_state.confirm_delete and st.session_state.delete_session_key:
                st.warning(f"Are you sure you want to delete session '{st.session_state.delete_session_key}'? This action cannot be undone.")
                delete_cols = st.columns(2)
                with delete_cols[0]:
                    if st.button("Confirm Delete", key="confirm_delete_button"):
                        if delete_session(db, st.session_state.delete_session_key):
                            st.session_state.current_session = None
                            st.session_state.delete_session_key = None
                            st.session_state.confirm_delete = False
                            st.rerun()
                with delete_cols[1]:
                    if st.button("Cancel", key="cancel_delete_button"):
                        st.session_state.delete_session_key = None
                        st.session_state.confirm_delete = False

            if st.session_state.current_session and st.session_state.current_session in session_options:
                selected_session = session_options[st.session_state.current_session]
                selected_docs = selected_session.selected_docs
                st.session_state.chat_sessions[st.session_state.current_session] = selected_docs

                if selected_docs:
                    st.session_state.vectorstore = load_vector_store(selected_docs[0], EMBEDDER)
                    db_toc = db.query(DocumentTOC).filter(DocumentTOC.doc_name == selected_docs[0]).first()
                    if db_toc:
                        st.session_state.doc_toc = db_toc.toc_items
                    else:
                        st.session_state.doc_toc = None
                else:
                    st.session_state.vectorstore = None
                    st.session_state.doc_toc = None

            elif session_keys:
                st.info("Select a chat session.")
            else:
                st.info("Create a chat session to start.")

        st.markdown("## Options")
        internet_search = st.toggle("Enable Internet Search", value=False)

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.session_state.current_session:
            with st.expander(f"Session Information: {st.session_state.current_session}"):
                selected_docs = st.session_state.chat_sessions.get(st.session_state.current_session, [])
                if selected_docs:
                    st.markdown("**Selected Documents:**")
                    for doc in selected_docs:
                        st.markdown(f"- {doc}")

            if st.session_state.elements is not None:
                st.markdown("**Images:**")
                for element in st.session_state.elements:
                    if isinstance(element, Image.Image):
                        st.image(element)
                    else:
                        st.info("Image can not load")

            messages_key = f"{st.session_state.current_session}_messages"

            if messages_key not in st.session_state:
                st.session_state[messages_key] = []

            with SessionLocal() as db:
                db_messages = db.query(ChatMessage).filter(ChatMessage.session_key == st.session_state.current_session).all()
                st.session_state[messages_key] = [{"role": msg.role, "content": msg.content} for msg in db_messages]

            for message in st.session_state[messages_key]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            is_searching = False
            if st.session_state.vectorstore is not None:
                qa_chain = init_chatbot(st.session_state.vectorstore, internet_search=internet_search)

                with st.container():
                    st.markdown(
                        """
                        <style>
                        .fixed-bottom {
                            position: fixed;
                            bottom: 0;
                            left: 0;
                            width: 100%;
                            background-color: #f0f2f6;
                            padding: 10px;
                            z-index: 1000;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.container():
                        if query := st.chat_input("Ask a question about the document"):
                            with SessionLocal() as db:
                                user_message = ChatMessage(session_key=st.session_state.current_session, role="user", content=query)
                                db.add(user_message)
                                db.commit()

                            st.session_state[messages_key].append({"role": "user", "content": query})
                            with st.chat_message("user"):
                                st.markdown(query)

                            with SessionLocal() as db:
                                if st.session_state.internet_search:
                                    with st.spinner("Searching internet"):
                                        answer = qa_chain.invoke({"input": query})
                                else:
                                    answer = qa_chain.invoke({"input": query})
                                if hasattr(answer, 'content'):
                                    st.session_state[messages_key].append({"role": "assistant", "content": answer.content})
                                    with st.chat_message("assistant"):
                                        st.markdown(answer.content)
                                    new_message = ChatMessage(session_key=st.session_state.current_session, role="assistant",
                                                              content=answer.content)
                                    db.add(new_message)
                                    db.commit()
                                else:
                                    st.session_state[messages_key].append({"role": "assistant", "content": str(answer)})
                                    with st.chat_message("assistant"):
                                        st.markdown(str(answer))
                                    new_message = ChatMessage(session_key=st.session_state.current_session, role="assistant",
                                                              content=str(answer))
                                    db.add(new_message)
                                    db.commit()

            else:
                st.info("Please load a document")

    with col2:
        if st.session_state.current_session and st.session_state.doc_toc and st.session_state.doc_chunks:

            st.header("Table of Contents")

            for i, item in enumerate(st.session_state.doc_toc):
                if item and isinstance(item, str):
                    toc_text = item

                    if toc_text in st.session_state.doc_chunks:
                        filename = st.session_state.doc_chunks[toc_text]
                        if filename and os.path.exists(os.path.join(CHUNK_FOLDER, filename)):
                            with st.expander(toc_text):
                                try:
                                    with open(os.path.join(CHUNK_FOLDER, filename), 'r', encoding='utf-8') as f:
                                        md_content = f.read()
                                    # Removed markdown.markdown conversion
                                    st.markdown(md_content, unsafe_allow_html=True) #Displaying the raw file content
                                except Exception as e:
                                    st.error(f"Error loading chunk file: {e}")
                        else:
                            st.warning(f"Chunk file not found for: {toc_text}")

if __name__ == "__main__":
    from database import Base, engine
    Base.metadata.create_all(engine)
    main()