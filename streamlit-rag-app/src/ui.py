import streamlit as st
from src.pdf_processor import process_pdf
from src.retriever import Retriever

def main():
    st.title("PDF Question Answering App")
    st.write("Ask questions based on the content of the PDF.")

    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file is not None:
        # Process the uploaded PDF
        full_text = process_pdf(pdf_file)
        retriever = Retriever(full_text)

        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if question:
                answer = retriever.retrieve_answer(question)
                st.write("Answer:", answer)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()