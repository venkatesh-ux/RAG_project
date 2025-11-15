import os
from typing import List, Optional, Union
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class VectorStore:
    def __init__(
        self,
        source: Optional[Union[FAISS, List[str]]] = None,
        vector_store_path: str = "faiss_vector_store",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        source: either a LangChain FAISS vectorstore instance OR a list of text chunks
                or an already-built retriever (object with .invoke()).
        """
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model

        # Two separate attributes so we don't confuse a retriever with a vectorstore
        self.vector_store: Optional[FAISS] = None
        self.retriever = None

        if source is None:
            # Try to load a vectorstore from disk (if present)
            if os.path.exists(self.vector_store_path):
                try:
                    emb = OpenAIEmbeddings(model=self.embedding_model)
                    self.vector_store = FAISS.load_local(self.vector_store_path, emb)
                except Exception:
                    # failed to load; leave both as None
                    self.vector_store = None
        elif isinstance(source, list):
            # List of text chunks -> build index
            emb = OpenAIEmbeddings(model=self.embedding_model)
            self.vector_store = FAISS.from_texts(source, emb)
        else:
            # source is an object. Decide if it's a vectorstore or a retriever
            # Vectorstore-like: has as_retriever AND save_local
            if hasattr(source, "as_retriever") and hasattr(source, "save_local"):
                self.vector_store = source
            # Retriever-like: has invoke() (new LangChain Runnable retriever)
            elif hasattr(source, "invoke"):
                self.retriever = source
            else:
                raise ValueError("Unsupported source type for VectorStore")

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        vector_store_path: str = "faiss_vector_store",
        embedding_model: str = "text-embedding-3-small",
    ):
        return cls(source=texts, vector_store_path=vector_store_path, embedding_model=embedding_model)

    @classmethod
    def load_local(
        cls,
        vector_store_path: str = "faiss_vector_store",
        embedding_model: str = "text-embedding-3-small",
    ):
        return cls(source=None, vector_store_path=vector_store_path, embedding_model=embedding_model)

    def save(self):
        """
        Save the vector store to disk.
        If only a retriever is present (no vectorstore), saving is not possible.
        """
        if self.vector_store is not None and hasattr(self.vector_store, "save_local"):
            self.vector_store.save_local(self.vector_store_path)
        else:
            raise RuntimeError("No vectorstore available to save. Create or load a vectorstore first.")

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Return a retriever object.
        - If this instance was initialized with an existing retriever, return it.
        - Otherwise convert the internal vectorstore to a retriever.
        """
        # If a retriever was supplied/created earlier, return it directly
        if self.retriever is not None:
            return self.retriever

        if self.vector_store is None:
            return None

        # Default search params
        search_kwargs = search_kwargs or {"k": 3}

        # Most LangChain vectorstores accept as_retriever(search_kwargs=...)
        # Do not pass unsupported args like `search_type` unless your vectorstore supports it.
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)