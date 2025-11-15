# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# # from src.embeddings import create_embeddings
# import os

# class EmbeddingManager:
#     def __init__(self, model_name="text-embedding-3-small"):
#         self.embeddings = OpenAIEmbeddings(model=model_name)

#     def create_embeddings(text):
#         """
#     Create embeddings for the given text by splitting it into chunks and embedding each chunk.

#     Args:
#         text (str): The input text to generate embeddings for.

#     Returns:
#         list: A list of embeddings for the text chunks.
#         """
#     # Split text into smaller chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
#         chunks = text_splitter.split_text(text)

#     # Generate embeddings for each chunk
#         embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
#         embeddings = embeddings_model.embed_documents(chunks)

    
#         return embeddings


#     def save_embeddings(self, texts, vector_store_path="faiss_vector_store"):
#         embeddings = self.create_embeddings(texts)
#         vector_store = FAISS.from_embeddings(embeddings)
#         vector_store.save_local(vector_store_path)

#     def load_embeddings(self, vector_store_path="faiss_vector_store"):
#         return FAISS.load_local(vector_store_path, self.embeddings)

# filepath: c:\Users\chven\OneDrive\Documents\GitHub\RAG_project\streamlit-rag-app\src\embeddings.py
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_embeddings(text):
    """
    Create embeddings for the given text by splitting it into chunks and embedding each chunk.

    Args:
        text (str): The input text to generate embeddings for.

    Returns:
        list: A list of embeddings for the text chunks.
    """
    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Generate embeddings for each chunk
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = embeddings_model.embed_documents(chunks)

    return embeddings
