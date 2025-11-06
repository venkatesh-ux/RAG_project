from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

class VectorStore:
    def __init__(self, vector_store_path="faiss_vector_store"):
        self.vector_store_path = vector_store_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = self.load_vector_store()

    def load_vector_store(self):
        if os.path.exists(self.vector_store_path):
            return FAISS.load_local(self.vector_store_path, self.embeddings)
        else:
            raise FileNotFoundError(f"Vector store not found at {self.vector_store_path}")

    def add_texts(self, texts):
        """Add new texts to the vector store."""
        self.vector_store.add_texts(texts)

    def query(self, query_text, k=3):
        """Query the vector store for similar texts."""
        return self.vector_store.similarity_search(query_text, k=k)

    def save(self):
        """Save the vector store to disk."""
        self.vector_store.save_local(self.vector_store_path)