import os

# Try to import streamlit; if it's not available (e.g., in a linter or non-Streamlit env),
# provide a minimal stub implementation so the script can run without import errors.
try:
    import streamlit as st
except Exception:
    class _StubSecrets:
        def get(self, key, default=None):
            return os.environ.get(key, default)

    class _StubStreamlit:
        secrets = _StubSecrets()

        def error(self, msg):
            print("ERROR:", msg)

        def stop(self):
            raise SystemExit("Stopped by stub Streamlit")

        def title(self, title_str):
            print(title_str)

        def text_input(self, prompt):
            try:
                # fallback to console input when not running in Streamlit
                return input(prompt + " ")
            except Exception:
                return ""

        def write(self, *args, **kwargs):
            print(*args, **kwargs)

    st = _StubStreamlit()

from src.pdf_processor import read_pdf
from src.retriever import Retriever

# ensure key is available BEFORE importing retriever
OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("OPENAI_API_KEY not set. Add it as an env var or in .streamlit/secrets.toml")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# Load PDF and prepare retriever
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


import streamlit as st

st.title("🎉 Streamlit Test App")
st.write("Streamlit is successfully detecting your app.py file.")