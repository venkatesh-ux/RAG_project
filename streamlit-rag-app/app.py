import streamlit as st
from src.pdf_processor import read_pdf
from src.retriever import Retriever

# Initialize the PDF reader and retriever
pdf_path = "data/books/sample.pdf"
full_text = read_pdf(pdf_path)
retriever = Retriever(full_text)

# Streamlit application title
st.title("PDF Question Answering App")

# User input for questions
user_question = st.text_input("Ask a question about the content of the PDF:")

if user_question:
    # Retrieve the answer based on the user's question
    answer = retriever.retrieve_answer(user_question)
    
    # Display the answer
    if answer:
        st.write("Answer:", answer)
    else:
        st.write("Sorry, I couldn't find an answer to your question.")