import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def init_chatbot(vector_store):
  llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # hoặc "gemini-pro"
    gemini_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0, # or other parameters
  )
  prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know. Don't try to make up an answer. Always respond concisely and only with a few sentences. Summarize the information in the context. Return the answer as a simple text string.\n\n{context}",
        ),
        ("human", "{input}"),
    ]
  )
  chain = create_retrieval_chain(vector_store.as_retriever(), prompt_template)
  return chain