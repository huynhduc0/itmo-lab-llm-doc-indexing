import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import chain
from langchain.tools import DuckDuckGoSearchRun

def init_chatbot(vector_store, internet_search=False):
    restrict = """
    You must base your answers primarily on the context provided from the document(s).

    If the context and search results don't contain information relevant to the question, simply say that you don't know. Do not make up answers.

    When using search results, incorporate them seamlessly and cite sources where appropriate.
    Always respond concisely and in a way that is easy to understand. Summarize the information, focusing on the main points and removing irrelevant details.
    Offer advice on how to use the information to help the user understand it better.
    Return the answer as a simple text string.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # hoặc "gemini-pro"
        gemini_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0, # or other parameters
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful and expert document assistant. You will be given a question, a context, and, if available, search results from the internet. Your job is to answer the question as accurately and helpfully as possible.

                """ +
                (""" You have access to the internet search results. Use them to add breadth and depth to your responses, providing more comprehensive and up-to-date information.""" if internet_search else "") +
                """\n\nContext:\n\n{context}""",
            ),
            ("human", "{input}"),
        ]
    )
    # Tạo retriever
    retriever = vector_store.as_retriever()

    # Tạo chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    if internet_search:
        search = DuckDuckGoSearchRun()
        def search_intent(x):
            query = x["input"]
            related_results = search.run(query)
            return related_results
        retrieval_chain = {"context": lambda x: format_docs(retriever.get_relevant_documents(x["input"])) + "\nWeb Search Results:\n" + search_intent(x), "input": RunnablePassthrough()} | prompt_template | llm
    else:
        retrieval_chain = {"context": lambda x: format_docs(retriever.get_relevant_documents(x["input"])), "input": RunnablePassthrough()} | prompt_template | llm

    return retrieval_chain

def generate_toc_with_llm(document_content):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # hoặc "gemini-pro"
        gemini_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0, # or other parameters
    )

    prompt = f"""
    Please generate a table of contents for the following document. The table of contents should be well-organized and reflect the main topics covered in the document. Return just list the headings.
    Document Content:
    {document_content}
    """
    try:
        toc = llm.invoke(prompt)
        return toc.split('\n')
    except Exception as e:
        print(f"Error generating TOC with LLM: {e}")
        return None