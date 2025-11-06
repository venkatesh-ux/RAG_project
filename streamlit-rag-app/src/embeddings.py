from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

class EmbeddingManager:
    def __init__(self, model_name="text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=model_name)

    def create_embeddings(self, texts):
        return self.embeddings.embed_documents(texts)

    def save_embeddings(self, texts, vector_store_path="faiss_vector_store"):
        embeddings = self.create_embeddings(texts)
        vector_store = FAISS.from_embeddings(embeddings)
        vector_store.save_local(vector_store_path)

    def load_embeddings(self, vector_store_path="faiss_vector_store"):
        return FAISS.load_local(vector_store_path, self.embeddings)