import os

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
    EMBEDDING_MODEL = "text-embedding-3-small"
    VECTOR_STORE_PATH = "faiss_vector_store"
    PDF_DIRECTORY = "data/books"
    CHUNK_SIZE = 2500
    CHUNK_OVERLAP = 200
    RETRIEVER_K = 3
    TEMPERATURE = 0.0