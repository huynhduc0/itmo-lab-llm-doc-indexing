import os
from langchain.llms import Gemini
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def init_chatbot(vector_store):
    llm = Gemini(model="models/gemini-pro",
                gemini_api_key=os.environ.get("GOOGLE_API_KEY")
    )
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    Question: {question}
    """
    PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
      )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
    )

    return qa_chain