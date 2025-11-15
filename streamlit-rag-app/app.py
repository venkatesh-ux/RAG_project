from src.pdf_processor import read_pdf
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

# Initialize vector_store in session_state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Button to process the PDF
if st.button("Process PDF"):
    try:
        full_text = read_pdf(pdf_path)
        st.success("PDF loaded successfully!")

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
        chunks = splitter.split_text(full_text)

        # Create vector store from chunks
        vector_store = VectorStore.from_texts(chunks, vector_store_path="faiss_vector_store")
        st.session_state.vector_store = vector_store  # Save to session_state
        st.success("Embeddings and vector store created successfully!")

        # Save the vector store explicitly
        vector_store.save()
        st.success("Vector store saved successfully!")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Disable the question field until the PDF is processed
if st.session_state.vector_store is None:
    st.info("Please process the PDF first to enable question answering.")
else:
    # Question input field (enabled only after processing the PDF)
    question = st.text_input("Ask a question about the PDF:")
    if question:
        try:
            retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            if retriever is None:
                st.error("Retriever not available. Vector store failed to initialize.")
            else:
                # Dynamically check for the correct method
                if hasattr(retriever, "retrieve"):
                    docs = retriever.retrieve(question)
                elif hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(question)
                else:
                    raise AttributeError("Retriever object has no method to retrieve documents.")

                # Display the retrieved context
                context = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)
                st.write("Answer Context:")
                st.write(context)
        except AttributeError as e:
            st.error(f"Error retrieving documents: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")