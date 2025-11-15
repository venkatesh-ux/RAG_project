from src.pdf_processor import read_pdf
from src.embeddings import create_embeddings
from src.vectorstore import VectorStore
import os
import streamlit as st

# Ensure OpenAI API key is set
OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("OPENAI_API_KEY not set. Add it as an environment variable or in .streamlit/secrets.toml")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# Streamlit app
st.title("PDF Question Answering App")

# Input for PDF path
pdf_path = st.text_input("PDF path", "C:/Users/chven/OneDrive/Documents/aaa_Books/Hands on machine learing book.pdf")

if pdf_path:
    try:
        # Read the PDF
        full_text = read_pdf(pdf_path)
        st.success("PDF loaded successfully!")
        
        # Create embeddings and vector store
        embeddings = create_embeddings(full_text)
        vector_store = VectorStore(embeddings)
        st.success("Embeddings and vector store created successfully!")
        
        # Input for user question
        question = st.text_input("Ask a question about the PDF:")
        if question:
            # Retrieve answer
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            st.write("Answer Context:")
            st.write(context)
    except ValueError as e:
        st.error(str(e))
    except FileNotFoundError as e:
        st.error(str(e))

# # Create embeddings and vector store
# embeddings = create_embeddings(full_text)
# vector_store = VectorStore(embeddings, vector_store_path="faiss_vector_store")
# st.success("Embeddings and vector store created successfully!")