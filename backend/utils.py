def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def search_intent(x, search):
    query = x["input"]
    related_results = search.run(query)
    return related_results
