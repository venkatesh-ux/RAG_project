# ...existing code...
import os

class Config:
    # do NOT put real keys here. Prefer environment or Streamlit secrets.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or None
    EMBEDDING_MODEL = "text-embedding-3-small"
    VECTOR_STORE_PATH = "faiss_vector_store"
    PDF_DIRECTORY = "data/books"
    CHUNK_SIZE = 2500
    CHUNK_OVERLAP = 200
    RETRIEVER_K = 3
    TEMPERATURE = 0.0

from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set in .env file.")