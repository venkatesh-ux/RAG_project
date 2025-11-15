# app.py
import re
from pathlib import Path
import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO

st.set_page_config(page_title="Local PDF Text Viewer - Safe Download", layout="wide")

st.title("Local PDF Text Viewer — Safe Download")
st.markdown(
    """
Upload a PDF and extract text locally. This app includes safe handling for
problematic Unicode surrogate characters so `st.download_button` won't crash.
No external APIs are used — everything runs on your machine.
"""
)

# -------------------------
# Helper functions
# -------------------------
_surrogate_re = re.compile(r'[\uD800-\uDFFF]')

def remove_surrogates(text: str) -> str:
    """Remove lone surrogate code points from text."""
    return _surrogate_re.sub("", text)

def text_to_bytes_safe(text: str, errors: str = "replace") -> bytes:
    """
    Convert text to UTF-8 bytes safely using the chosen error handler.
    errors: 'replace' | 'ignore' | 'xmlcharrefreplace' | 'strict'
    """
    return text.encode("utf-8", errors=errors)

def read_pdf_bytes(file_like) -> str:
    """
    Read PDF from a file-like object or path and extract text from all pages.
    Returns a single string with pages joined by two newlines.
    """
    try:
        reader = PdfReader(file_like)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages_text.append(text.strip())
    full_text = "\n\n".join(pages_text)
    return full_text

# -------------------------
# UI Controls
# -------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

st.sidebar.header("Download / Encoding options")
error_handling = st.sidebar.selectbox(
    "UTF-8 error handling for download (used when encoding to bytes)",
    options=["replace", "ignore", "xmlcharrefreplace", "strict"],
    index=0,
    help="How to handle characters that can't be encoded to UTF-8. 'replace' inserts �, 'ignore' drops them, 'xmlcharrefreplace' writes &#NNNN;, 'strict' will raise an error."
)

remove_surrogates_checkbox = st.sidebar.checkbox(
    "Remove surrogate code points before download (regex)", value=False,
    help="If checked, all lone surrogate code points (e.g. \\ud800-\\udfff) are removed before download."
)

save_uploaded_pdf = st.sidebar.checkbox("Enable 'Save PDF to disk' button", value=True)

# -------------------------
# Main flow
# -------------------------
if uploaded_file is not None:
    st.write(f"**File:** {uploaded_file.name} — {uploaded_file.size} bytes")

    # Ensure bytes-like for PdfReader
    # PyPDF2 accepts file-like; uploaded_file is a BytesIO-like object
    uploaded_file.seek(0)
    pdf_bytes = BytesIO(uploaded_file.read())
    full_text = read_pdf_bytes(pdf_bytes)

    if not full_text.strip():
        st.warning("No extractable text found in this PDF (it might be scanned images).")
        st.info("If it's a scanned PDF, consider using OCR (pytesseract + pdf2image). Tell me if you want an OCR option.")
    else:
        # Preview or full text
        st.markdown("### Preview / Display options")
        show_full = st.checkbox("Show full extracted text", value=False)
        if show_full:
            st.subheader("Full extracted text")
            st.text_area("Text", value=full_text, height=600)
        else:
            preview_chars = st.number_input(
                "Preview characters",
                min_value=100, max_value=200000, value=1000, step=100,
                help="Number of characters to show in the preview."
            )
            st.subheader("Preview of extracted text")
            st.text_area("Preview", value=full_text[:preview_chars], height=300)

        # Prepare cleaned/processed versions depending on user options
        processed_text = full_text
        if remove_surrogates_checkbox:
            processed_text = remove_surrogates(processed_text)

        # Convert to bytes with selected error handling
        try:
            safe_bytes = text_to_bytes_safe(processed_text, errors=error_handling)
        except Exception as e:
            st.error(f"Error encoding text with errors='{error_handling}': {e}")
            # fallback to replace
            safe_bytes = text_to_bytes_safe(processed_text, errors="replace")
            st.info("Falling back to errors='replace' for download bytes.")

        # Download buttons (use bytes to avoid Streamlit re-encoding issues)
        st.markdown("### Download extracted text")
        st.download_button(
            label="Download extracted text (.txt)",
            data=safe_bytes,
            file_name=Path(uploaded_file.name).stem + ".txt",
            mime="text/plain",
        )

        # Also provide an option to download a 'cleaned' variant if surrogates were removed
        if not remove_surrogates_checkbox:
            cleaned_text = remove_surrogates(full_text)
            cleaned_bytes = text_to_bytes_safe(cleaned_text, errors=error_handling)
            st.download_button(
                label="Download extracted text (surrogates removed).txt",
                data=cleaned_bytes,
                file_name=Path(uploaded_file.name).stem + "_cleaned.txt",
                mime="text/plain",
            )

        # Option: save the uploaded PDF locally as binary
        if save_uploaded_pdf:
            if st.button("Save uploaded PDF to local disk"):
                save_path = Path.cwd() / uploaded_file.name
                try:
                    # write the original uploaded bytes to disk (binary)
                    with open(save_path, "wb") as f:
                        uploaded_file.seek(0)
                        f.write(uploaded_file.read())
                    st.success(f"Saved uploaded PDF to {save_path}")
                except Exception as e:
                    st.error(f"Failed to save PDF: {e}")

        # Option: write extracted text to a local file using the same safe bytes
        if st.button("Save extracted text to local disk (.txt)"):
            out_name = Path(uploaded_file.name).stem + ".txt"
            out_path = Path.cwd() / out_name
            try:
                with open(out_path, "wb") as f:
                    f.write(safe_bytes)
                st.success(f"Extracted text saved to {out_path}")
            except Exception as e:
                st.error(f"Failed to save extracted text: {e}")

else:
    st.info("Upload a PDF to extract and view its text locally.")
