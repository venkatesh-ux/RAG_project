import os
import sys
from pathlib import Path
import streamlit as st
from PyPDF2 import PdfReader

# optional: add project root so config can be imported
sys.path.append(r"c:/Users/chven/OneDrive/Documents/GitHub/RAG_project")
import config  # ensure this file exists at the path above and defines API_KEY

# set API key for openai use
os.environ["OPENAI_API_KEY"] = config.API_KEY

# Langchain imports for embeddings + FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import openai  # pip install openai

openai.api_key = config.API_KEY

st.title("PDF Question Answering App (fixed)")

def read_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages_text)

def split_text_into_chunks(text: str, chunk_size=2500, chunk_overlap=200):
    # very small splitter fallback (try simple slicing) if RecursiveCharacterTextSplitter not available
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def create_vector_store(chunks, vector_store_path="faiss_vector_store"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # FAISS.from_texts expects a list of strings
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(vector_store_path)
    return vector_store

def retrieve_context(vector_store, question: str, k=3):
    # similarity_search is commonly available and returns Document objects with .page_content
    docs = vector_store.similarity_search(question, k=k)
    # join retrieved chunk texts
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

def generate_answer_openai(context: str, question: str) -> str:
    # instruct model to answer ONLY from context and return verbatim if present
    system_prompt = "You are a helpful assistant. Answer ONLY from the provided context. If the answer appears verbatim in the context, return that exact sentence or phrase. If insufficient, say 'I don't know.'"
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer (use only the context):"

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=256,
    )
    # extract text
    return resp["choices"][0]["message"]["content"].strip()

# UI
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    pdf_path = Path("uploaded_file.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    full_text = read_pdf(pdf_path)
    st.write("### PDF preview (first 1000 chars):")
    st.text(full_text[:1000])

    # split and build vector store
    chunks = split_text_into_chunks(full_text)
    st.write(f"Created {len(chunks)} chunks.")
    vector_store = create_vector_store(chunks)

    question = st.text_input("Ask a question about the PDF:")
    if question:
        context = retrieve_context(vector_store, question, k=3)
        st.write("### Retrieved context preview:")
        st.text(context[:1000])
        answer = generate_answer_openai(context, question)
        st.write("### Answer:")
        st.write(answer)
