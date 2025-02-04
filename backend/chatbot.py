import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_core.runnables import RunnablePassthrough 

def init_chatbot(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # hoặc "gemini-pro"
        gemini_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0, # or other parameters
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful and expert document assistant. Use the provided information to answer the user's question as accurately and helpfully as possible.
                If you don't know the answer, simply say that you don't know. Do not make up answers.
                Always respond concisely and in a way that is easy to understand, using no more than a few sentences. Summarize the information from the context, focusing on the main points and removing irrelevant details.
                Offer advice on how to use the information in the document to help the user understand it better.
                Return the answer as a simple text string.

                Context:\n\n{context}""",
            ),
            ("human", "{input}"),
        ]
    )
    # Tạo retriever
    retriever = vector_store.as_retriever()

    # Tạo chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_answer(input, context):
        prompt = prompt_template.format(input=input, context=context)
        return llm.invoke(prompt).content

    retrieval_chain = RunnablePassthrough.assign(context=lambda x: format_docs(retriever.get_relevant_documents(x["input"]))) | prompt_template | llm

    return retrieval_chain