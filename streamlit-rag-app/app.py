from src.pdf_processor import read_pdf
from src.embeddings import create_embeddings
from src.vectorstore import VectorStore
import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        full_text = read_pdf(pdf_path)
        st.success("PDF loaded successfully!")

        # Split text into chunks (pass strings to VectorStore)
        splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
        chunks = splitter.split_text(full_text)

        # Create/load vector store from chunks
        vector_store = VectorStore.from_texts(chunks, vector_store_path="faiss_vector_store")
        st.success("Embeddings and vector store created successfully!")

        # User question...
        question = st.text_input("Ask a question about the PDF:")
        if question:
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            if retriever is None:
                st.error("Retriever not available. Vector store failed to initialize.")
            else:
                try:
                    # Use the correct method for retrieving documents
                    docs = retriever.get_relevant_documents(question)
                    context = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)
                    st.write("Answer Context:")
                    st.write(context)
                except AttributeError as e:
                    st.error(f"Error retrieving documents: {e}")
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except ValueError as e:
        st.error(f"Error processing the PDF: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# # Create embeddings and vector store
# embeddings = create_embeddings(full_text)
# vector_store = VectorStore(embeddings, vector_store_path="faiss_vector_store")
# st.success("Embeddings and vector store created successfully!")