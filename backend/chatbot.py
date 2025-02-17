import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import chain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage

def init_chatbot(vector_store, internet_search=False):
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

                You must primarily base your answers on the context provided from the document(s).

                If the context and search results don't contain information relevant to the question, simply say that you don't know. Do not make up answers.

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
    print(f"Initializing LLM for TOC generation")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # hoặc "gemini-pro"
        gemini_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0, # or other parameters
    )
    print(f"Generating TOC with LLM for document")
    prompt = f"""
        Extract and generate a structured table of contents based on the key topics in the document. 
        Return only the actual headings or section titles that represent meaningful content, one per line, without numbering or bullet points.

        Exclude any generic, placeholder, or irrelevant headings such as "Title", "Table of Contents", "Document Title", or similar sections. Only include headings that are directly related to the document's main content and structure.

        Do not infer or create a "Title" if it is not explicitly mentioned as a heading in the document. Focus solely on sections that provide clear, substantive content.

        If the document lacks clear headings, infer a few key topics based on the content to summarize the document effectively.

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