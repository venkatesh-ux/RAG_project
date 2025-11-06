from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
import os
import config

import os
from src.config import Config

# prefer environment variable, fall back to Config
OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or Config.OPENAI_API_KEY
if not OPENAI_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set. Add it as an env var or in .streamlit/secrets.toml")
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# import langchain LLMs after the key is set
try:
    from langchain.chat_models import ChatOpenAI
except Exception:
    # Fallback import may not be available in all langchain versions or environments;
    # set ChatOpenAI to None so callers can detect lack of an LLM implementation.
    ChatOpenAI = None

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