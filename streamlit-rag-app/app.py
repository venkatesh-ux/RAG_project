# app.py
import os
from pathlib import Path
import streamlit as st
from PyPDF2 import PdfReader

st.set_page_config(page_title="Local PDF Text Viewer", layout="wide")

st.title("Local PDF Text Viewer (No external calls)")

st.markdown(
    """
Upload a PDF and this app will extract and display the text locally.
No OpenAI / LangChain / FAISS / external APIs are used.
Streamlit runs on localhost (by default http://localhost:8501).
"""
)

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

def read_pdf_bytes(pdf_bytes):
    """
    Read PDF from bytes and extract text from all pages.
    Returns a single string with pages joined by newlines.
    """
    reader = PdfReader(pdf_bytes)
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        # Optionally strip leading/trailing whitespace per page
        pages_text.append(text.strip())
    full_text = "\n\n".join(pages_text)
    return full_text

if uploaded_file is not None:
    # Show filename and size
    st.write(f"**File:** {uploaded_file.name} — {uploaded_file.size} bytes")

    # Read & extract text
    full_text = read_pdf_bytes(uploaded_file)

    if not full_text.strip():
        st.warning("No extractable text found in this PDF (it might be scanned images).")
        st.info("If it's a scanned PDF, consider OCR (tesseract, pytesseract) before text extraction.")
    else:
        # Option: show only a preview or full text
        show_full = st.checkbox("Show full extracted text", value=False)
        if show_full:
            st.subheader("Full extracted text")
            st.text_area("Text", value=full_text, height=600)
        else:
            # Show first N characters as a preview
            preview_chars = st.number_input("Preview characters", min_value=100, max_value=20000, value=1000, step=100)
            st.subheader("Preview of extracted text")
            st.text_area("Preview", value=full_text[:preview_chars], height=300)

        # Download button for extracted text
        st.download_button(
            label="Download extracted text (.txt)",
            data=full_text,
            file_name=Path(uploaded_file.name).stem + ".txt",
            mime="text/plain",
        )

    # Optionally store the uploaded file locally (temporary)
    if st.button("Save uploaded PDF to local disk"):
        save_path = Path.cwd() / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved to {save_path}")

else:
    st.info("Upload a PDF to extract and view its text locally.")
