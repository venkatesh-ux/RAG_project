import os
from typing import List, Optional, Union
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class VectorStore:
    def __init__(self, source: Optional[Union[FAISS, List[str]]] = None, vector_store_path: str = "faiss_vector_store", embedding_model: str = "text-embedding-3-small"):
        """
        source: either a LangChain FAISS vectorstore instance OR a list of text chunks
        """
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.vector_store = None

        if source is None:
            # Try to load from disk
            if os.path.exists(self.vector_store_path):
                try:
                    self.vector_store = FAISS.load_local(self.vector_store_path, OpenAIEmbeddings(model=self.embedding_model))
                except Exception:
                    self.vector_store = None
        elif hasattr(source, "as_retriever"):
            # Already a LangChain vectorstore
            self.vector_store = source
        elif isinstance(source, list):
            # List of text chunks -> build index
            emb = OpenAIEmbeddings(model=self.embedding_model)
            self.vector_store = FAISS.from_texts(source, emb)
        else:
            raise ValueError("Unsupported source type for VectorStore")

    @classmethod
    def from_texts(cls, texts: List[str], vector_store_path: str = "faiss_vector_store", embedding_model: str = "text-embedding-3-small"):
        return cls(source=texts, vector_store_path=vector_store_path, embedding_model=embedding_model)

    @classmethod
    def load_local(cls, vector_store_path: str = "faiss_vector_store", embedding_model: str = "text-embedding-3-small"):
        return cls(source=None, vector_store_path=vector_store_path, embedding_model=embedding_model)

    def save(self):
        """
        Save the vector store to disk.
        """
        if self.vector_store is not None:
            self.vector_store.save_local(self.vector_store_path)

    def as_retriever(self, search_type: str = "similarity", search_kwargs: Optional[dict] = None):
        if self.vector_store is None:
            return None
        search_kwargs = search_kwargs or {"k": 3}
        return self.vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)