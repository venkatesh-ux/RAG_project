import os
from pathlib import Path
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import sys
sys.path.append("c:/Users/chven/OneDrive/Documents/GitHub/RAG_project")
import config

# Set OpenAI API key
import config
os.environ["OPENAI_API_KEY"] = config.API_KEY

# Step 1: PDF Ingestion
def read_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n\n".join(pages_text)
    return full_text

# Step 2: Text Splitting
def split_text(full_text, chunk_size=2500, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(full_text)
    return chunks

# Step 3: Embedding Creation and Vector Store
def create_vector_store(chunks, vector_store_path="faiss_vector_store"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(vector_store_path)
    return vector_store

# Step 4: Retrieval
def retrieve_answer(vector_store, question, k=3):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

# Step 5: Answer Generation
def generate_answer(context, question):
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided context.
        If the context is insufficient, just say you don't know.
        
        {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    final_prompt = prompt.invoke({'context': context, 'question': question})
    answer = llm.invoke(final_prompt)
    return answer

# Streamlit App
st.title("PDF Question Answering App")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    pdf_path = Path("uploaded_file.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 1: Read PDF
    full_text = read_pdf(pdf_path)
    st.write("### PDF Content:")
    st.write(full_text[:1000])  # Display the first 1000 characters of the PDF content

    # Step 2: Split Text
    chunks = split_text(full_text)

    # Step 3: Create Vector Store
    vector_store = create_vector_store(chunks)

    # Question input
    question = st.text_input("Ask a question about the PDF:")

    if question:
        # Step 4: Retrieve Answer
        context = retrieve_answer(vector_store, question)

        # Step 5: Generate Answer
        answer = generate_answer(context, question)

        # Display the answer
        st.write("### Answer:")
        st.write(answer.content)