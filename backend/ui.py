import streamlit as st
import os
from rag_utils import load_document, create_vector_store
from chatbot import init_chatbot

def main():
    st.title("Docs Assistant")

    uploaded_file = st.file_uploader("Upload a document")
    url = st.text_input("Enter a URL")
    if uploaded_file is not None:
        with open("temp_file", "wb") as f:
          f.write(uploaded_file.getbuffer())
        document = load_document(file_path="temp_file")
        os.remove("temp_file")
    elif url:
        document = load_document(url=url)
    else:
        document = None

    if document:
        vector_store = create_vector_store(document)
        qa_chain = init_chatbot(vector_store)

        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if query := st.chat_input("Ask a question about the document"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                answer = qa_chain.run(query)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
if __name__ == '__main__':
    main()