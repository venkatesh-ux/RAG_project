from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
import os
import config

class Retriever:
    def __init__(self, vector_store_path):
        os.environ["OPENAI_API_KEY"] = config.API_KEY
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)

    def retrieve(self, query, k=3):
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query)
        return retrieved_docs

    def format_retrieved_docs(self, retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs) if retrieved_docs else "No relevant documents found."