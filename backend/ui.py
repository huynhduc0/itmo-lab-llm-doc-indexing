import streamlit as st
import os
from rag_utils import load_document, create_vector_store, load_vector_store, save_uploaded_file  # Import save_uploaded_file
from chatbot import init_chatbot, generate_toc_with_llm
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from bs4 import BeautifulSoup
import requests
import pdfplumber
import docx2txt
import dotenv
from PIL import Image
from database import SessionLocal, ChatSession, ChatMessage, DocumentInfo, DocumentTOC  # Import database classes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage  # Added import
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

dotenv.load_dotenv()

# Khởi tạo Embeddings (hoặc GeminiEmbeddings nếu bạn muốn)
EMBEDDER = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
UPLOAD_FOLDER = "uploaded_documents"  # Thư mục lưu trữ tài liệu

# Database configuration from environment variables
DB_TYPE = os.getenv("DB_TYPE", "sqlite")  # Default to SQLite if not specified
DB_URI = os.getenv("DB_URI", "sqlite:///./doc_assistant.db")  # SQLite default

# Create engine based on DB_TYPE
engine = create_engine(DB_URI)

#Create database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_available_docs():
    return [f for f in os.listdir('document_indexes') if not f.startswith('.') and not os.path.isfile(os.path.join('document_indexes', f))]


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
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url  # Add https:// if no scheme is present
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all heading tags (h1, h2, h3, h4, h5', 'h6'])
        heading_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        toc = []
        for tag in heading_tags:
            toc.append({"text": tag.text, "id": tag.get('id')})
        return toc
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return None


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


def generate_toc_with_llm(document_content):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # hoặc "gemini-pro"
        gemini_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0,  # or other parameters
    )

    prompt = f"""
    Please generate a table of contents for the following document. The table of contents should be well-organized and reflect the main topics covered in the document. Return just a list of the headings, one heading per line.  Do not include any numbering or bullet points. Do not include "Title" as a heading unless it is explicitly provided in the document. If there are no clear headings, generate a few key topics that summarize the document's content.
    Document Content:
    {document_content}
    """
    try:
        toc = llm.invoke(prompt)
        print(f"Generated TOC: {toc}")
        if isinstance(toc, str):
            return toc.split('\n')
        elif isinstance(toc, BaseMessage):
            print(f"Generated TOC: {toc.content}")
            return toc.content.split('\n')
        else:
            print(f"Unexpected TOC format: {type(toc)}")
            return None
    except Exception as e:
        print(f"Error generating TOC with LLM: {e}")
        return None


def process_document(db, file=None, url=None):
    """Load document and create vectorstore."""
    if not file and not url:
        st.warning('Please load a document or enter a URL.')
        return None

    with st.spinner():
        if file:
            # Lưu file tải lên
            file_path = save_uploaded_file(file)
            doc_name = file.name[:file.name.index('.')]
            st.session_state.doc_name = doc_name
            st.session_state.doc_url = None

            try:  # New
                elements = load_document(file_path=file_path)
                # OCR
                # Check and load file
                vectorstore = create_vector_store(elements, doc_name, EMBEDDER)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.success('Vector store successfully ✔️')
                    file_extension = os.path.splitext(file_path)[1].lower()

                    if file_extension == ".pdf":
                        toc = get_table_of_contents_pdf(file_path)
                        if not toc:
                            doc_content = load_document(file_path=file_path)
                            toc = generate_toc_with_llm(doc_content)
                    elif file_extension == ".docx" or file_extension == ".doc":
                        toc = get_table_of_contents_docx(file_path)
                    else:
                        toc = None

                    # Save TOC to database
                    if toc:
                        existing_toc = db.query(DocumentTOC).filter(DocumentTOC.doc_name == doc_name).first()
                        if existing_toc:
                            existing_toc.toc_items = toc
                        else:
                            db_toc = DocumentTOC(doc_name=doc_name, toc_items=toc)
                            db.add(db_toc)
                        db.commit()
                        st.session_state.doc_toc = toc  # Store in session state after saving to db
                    else:
                        st.session_state.doc_toc = None  # or some default value to indicate no toc.

                return  # Return to the next state to update or load.

            except Exception as e:  # New
                db.rollback()
                st.error(f"Error loading {file_path}: {e}")  # New
                return None

        elif url:  # Nếu có URL
            st.session_state.doc_url = url
            doc = load_document(url=url)
            doc_name = url.replace("/", "_").replace(":", "_")
            st.session_state.doc_name = doc_name
            st.session_state.vectorstore = create_vector_store(doc, doc_name, EMBEDDER)
        st.success('Document load successfully ✔️')
        return st.session_state.vectorstore


def display_messages(messages_key):
    """Displays the chat messages."""
    if messages_key in st.session_state:
        for message in st.session_state[messages_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def show_document_section(selected_toc_item):
    """Displays the document section content."""
    if st.session_state.vectorstore:
        # Assuming your vectorstore has a way to search for similar content
        # This is a VERY basic example and may not work directly

        if isinstance(selected_toc_item, str):  # PDF Case
            results = st.session_state.vectorstore.similarity_search(selected_toc_item, k=3)  # Get top 3 matches
        else:  # Web Scraped Case (assuming you have the ID)
            results = st.session_state.vectorstore.similarity_search(selected_toc_item["text"], k=3)  # Get top 3 matches

        if results:
            for doc in results:
                st.write(doc.page_content)  # Display content of matched documents

        else:
            st.write("No matching content found.")

    else:
        st.write("No document loaded.")


def get_session_options(db):
    """Fetch chat sessions from the database."""
    sessions = db.query(ChatSession).all()
    return {session.session_key: session for session in sessions}  # Return a dict

def delete_session(db, session_key):
    """Deletes a chat session and its messages from the database."""
    try:
        # Get the session to delete
        session_to_delete = db.query(ChatSession).filter(ChatSession.session_key == session_key).first()

        if session_to_delete:
            # Delete associated messages first
            db.query(ChatMessage).filter(ChatMessage.session_key == session_key).delete()

            # Delete the session
            db.delete(session_to_delete)

            db.commit()
            st.success(f"Session '{session_key}' and its messages deleted successfully!")
            return True  # Indicate successful deletion
        else:
            st.warning(f"Session '{session_key}' not found.")
            return False  # Indicate session not found
    except Exception as e:
        db.rollback()
        st.error(f"Error deleting session '{session_key}': {e}")
        return False  # Indicate deletion failure


def main():
    # Initialize session state
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
    if 'chat_mode' not in st.session_state:
        st.session_state.chat_mode = "File hiện tại"
    if 'deep_reasoning' not in st.session_state:
        st.session_state.deep_reasoning = False
    if 'internet_search' not in st.session_state:
        st.session_state.internet_search = False
    if 'elements' not in st.session_state:
        st.session_state.elements = None
    if 'modal_open' not in st.session_state:
        st.session_state.modal_open = None
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

    # Main sidebar for main controls
    with st.sidebar:
        st.title('DocAssistant')

        st.markdown('## New Document')
        file = st.file_uploader(label='')
        url = st.text_input("Enter a URL", key="url_input")  # Thêm ô nhập URL
        applied = st.button(label='📄 Load Document')  # Button Load Document

        # if apply button clicked
        if applied:
            with SessionLocal() as db:
                try:  # To manage exceptions.
                    st.session_state.vectorstore = process_document(db=db, file=file, url=url)
                except Exception as e:
                    db.rollback()  # Rollback if there was an error.
                    st.error(f"Error during document loading: {e}")  # Display
                finally:
                    db.close()  # close the database to free resource.

        st.markdown('## Chat Sessions')
        with st.form("new_session_form", clear_on_submit=True):
            session_name = st.text_input("Session Name", "Session")
            selected_docs = st.multiselect("Select Documents", get_available_docs(), key="selected_docs")
            create_session = st.form_submit_button("Create New Session")

            if create_session:
                with SessionLocal() as db:
                    session_key = session_name.replace(" ", "_")  # Tạo key từ tên session
                    if session_key in st.session_state.chat_sessions:
                        st.warning(f"Session '{session_name}' already exists. Please use a different name.")
                    else:
                        st.session_state.chat_sessions[session_key] = selected_docs
                        db_session = ChatSession(session_key=session_key, selected_docs=selected_docs)
                        db.add(db_session)
                        db.commit()
                        st.session_state.current_session = session_key

                        # Load TOC and Vectorstore of file đầu tiên trong danh sách
                        if selected_docs:
                            # Load vector store của file đầu tiên trong danh sách
                            st.session_state.vectorstore = load_vector_store(selected_docs[0], EMBEDDER)

                            # Load TOC from the database
                            db_toc = db.query(DocumentTOC).filter(DocumentTOC.doc_name == selected_docs[0]).first()
                            if db_toc:
                                st.session_state.doc_toc = db_toc.toc_items
                            else:
                                st.session_state.doc_toc = None

                        else:
                            st.session_state.vectorstore = None  # Set về None nếu không có file nào được chọn
                            st.session_state.doc_toc = None

        # Database configuration
        DB_TYPE = os.getenv("DB_TYPE", "sqlite")  # Default to SQLite
        DB_URI = os.getenv("DB_URI", "sqlite:///./doc_assistant.db")  # SQLite default

        with SessionLocal() as db:
            session_options = get_session_options(db)
            session_keys = list(session_options.keys())

            # Collapsible session list
            with st.expander("Session List", expanded=True):
                for session_key in session_keys:
                    cols = st.columns([0.7, 0.3])  # Adjust column ratio as needed
                    with cols[0]:
                        if st.button(session_key, key=f"session_{session_key}"):
                            st.session_state.current_session = session_key
                    with cols[1]:
                         if st.button("Delete", key=f"delete_{session_key}"): # Use normal button
                            st.session_state.delete_session_key = session_key
                            st.session_state.confirm_delete = True

            # Confirmation dialog for deletion
            if st.session_state.confirm_delete and st.session_state.delete_session_key:
                st.warning(f"Are you sure you want to delete session '{st.session_state.delete_session_key}'? This action cannot be undone.")
                delete_cols = st.columns(2)
                with delete_cols[0]:
                    if st.button("Confirm Delete", key="confirm_delete_button"):
                        if delete_session(db, st.session_state.delete_session_key):
                            # Reset states after successful deletion
                            st.session_state.current_session = None
                            st.session_state.delete_session_key = None
                            st.session_state.confirm_delete = False
                            # No longer need to clear query params or rerun
                            st.rerun()  # Refresh the list
                with delete_cols[1]:
                    if st.button("Cancel", key="cancel_delete_button"):
                        st.session_state.delete_session_key = None
                        st.session_state.confirm_delete = False
                         # No longer need to clear query params or rerun

            # Load session data if a session is selected
            if st.session_state.current_session and st.session_state.current_session in session_options:
                selected_session = session_options[st.session_state.current_session]
                selected_docs = selected_session.selected_docs
                st.session_state.chat_sessions[st.session_state.current_session] = selected_docs

                # Load vector store and TOC
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
                st.info("Select a chat session.") #If there's no current session, and session keys, tell the user to select a session
            else:
                st.info("Create a chat session to start.")

        st.markdown("## Options")
        internet_search = st.toggle("Enable Internet Search", value=False)

    # Main Chat Interface
    col1, col2 = st.columns([3, 1])  # Adjust column ratios

    with col1:
        if st.session_state.current_session:  # Đảm bảo đã chọn session rồi
            with st.expander(f"Session Information: {st.session_state.current_session}"):
                # Show selected documents
                selected_docs = st.session_state.chat_sessions.get(st.session_state.current_session, [])
                if selected_docs:
                    st.markdown("**Selected Documents:**")
                    for doc in selected_docs:
                        st.markdown(f"- {doc}")

            # Display Document Section here, right after Session Info
            if st.session_state.show_doc_section:
                st.markdown("---")  # Add a visual separator
                st.markdown(f"### Document Section: {st.session_state.selected_toc_item}")
                show_document_section(st.session_state.selected_toc_item)

                if st.button("Close Document Section"):
                    st.session_state.show_doc_section = False
                    st.session_state.selected_toc_item = None

            # Main chat area
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

            # Load messages from the database:
            with SessionLocal() as db:
                db_messages = db.query(ChatMessage).filter(ChatMessage.session_key == st.session_state.current_session).all()
                st.session_state[messages_key] = [{"role": msg.role, "content": msg.content} for msg in db_messages]

            for message in st.session_state[messages_key]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            is_searching = False
            if st.session_state.vectorstore is not None:
                qa_chain = init_chatbot(st.session_state.vectorstore, internet_search=internet_search)

                # Adding st.container to pin the chat input
                with st.container():
                    st.markdown(
                        """
                        <style>
                        .fixed-bottom {
                            position: fixed;
                            bottom: 0;
                            left: 0;
                            width: 100%;
                            background-color: #f0f2f6; /* Adjust color as needed */
                            padding: 10px;
                            z-index: 1000; /* Ensure it's on top of other elements */
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Put chat input into the container
                    with st.container():
                        if query := st.chat_input("Ask a question about the document"):  # Kiểm tra lại chỗ này
                           # Save user message to the database
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
                                    st.session_state[messages_key].append({"role": "assistant", "content": answer.content})  # Save message
                                    st.chat_message("assistant").markdown(answer.content)
                                    new_message = ChatMessage(session_key=st.session_state.current_session, role="assistant",
                                                              content=answer.content)
                                    db.add(new_message)
                                    db.commit()
                                else:
                                    st.session_state[messages_key].append({"role": "assistant", "content": str(answer)})  # Save message
                                    st.chat_message("assistant").markdown(str(answer))
                                    new_message = ChatMessage(session_key=st.session_state.current_session, role="assistant",
                                                              content=str(answer))
                                    db.add(new_message)
                                    db.commit()

            else:
                st.info("Please load a document")


    with col2:
        show_toc()

def show_toc():
    """Displays the table of contents."""
    st.header("Table of Contents")
    if st.session_state.current_session and st.session_state.doc_toc:
        with st.expander("Table of Contents", expanded=True):
            for i, item in enumerate(st.session_state.doc_toc):
                if item and isinstance(item, str):
                    toc_text = item
                    if st.button(toc_text, key=f"toc_{i}_main"):
                        st.session_state.show_doc_section = True
                        st.session_state.selected_toc_item = toc_text
    else:
        st.info("No document loaded or no table of contents available for this session.")


if __name__ == "__main__":
    # This is to ensure tables are created on app startup
    from database import Base, engine
    Base.metadata.create_all(engine)
    main()