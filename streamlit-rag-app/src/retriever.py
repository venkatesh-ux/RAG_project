import os
from src.config import Config

# ensure API key is available before importing langchain LLMs
OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or Config.OPENAI_API_KEY
if not OPENAI_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set. Add it as an env var or in .streamlit/secrets.toml")
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

# import a chat LLM with safe fallbacks
try:
    from langchain.chat_models import ChatOpenAI
except Exception:
    try:
        import importlib
        llms_mod = importlib.import_module("langchain.llms")
        ChatOpenAI = getattr(llms_mod, "OpenAI", None)
    except Exception:
        ChatOpenAI = None

class Retriever:
    def __init__(self, vector_store_path):
        embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        # load the saved FAISS vector store
        self.vector_store = FAISS.load_local(vector_store_path, embeddings)

        if ChatOpenAI is None:
            raise ImportError("No compatible ChatOpenAI/OpenAI LLM found in installed langchain.")
        self.llm = ChatOpenAI(temperature=Config.TEMPERATURE)

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant. Answer ONLY from the provided context.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            ),
        )

    def retrieve_documents(self, query, k=None):
        k = k or Config.RETRIEVER_K
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        # prefer get_relevant_documents() to obtain Document objects
        docs = retriever.get_relevant_documents(query)
        return docs

    def retrieve_answer(self, query, k=None):
        docs = self.retrieve_documents(query, k)
        context = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs) if docs else ""
        # lazy import of LLMChain to handle langchain version differences
        try:
            import importlib
            try:
                chains_mod = importlib.import_module("langchain.chains")
                LLMChain = getattr(chains_mod, "LLMChain")
            except Exception:
                chains_mod = importlib.import_module("langchain.chains.llm")
                LLMChain = getattr(chains_mod, "LLMChain")
        except Exception:
            raise ImportError("LLMChain not available in installed langchain.")
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return chain.run({"context": context, "question": query})
