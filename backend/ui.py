import streamlit as st
import os
from rag_utils import load_document, create_vector_store
from chatbot import init_chatbot
import json
def main():
    st.title("Docs Assistant")

    uploaded_file = st.file_uploader("Upload a document")
    url = st.text_input("Enter a URL")
    temp_file_path = "temp_file"
    if uploaded_file is not None:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        document = load_document(file_path=temp_file_path)
        os.remove(temp_file_path)
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
            answer = qa_chain.invoke({"input": query})
            st.markdown(answer["answer"]) # Hiển thị answer trực tiếp
            print(answer["answer"].) # Hiển thị answer trong terminal
            st.session_state.messages.append({"role": "assistant", "content": answer["answer"]})
if __name__ == '__main__':
    main()