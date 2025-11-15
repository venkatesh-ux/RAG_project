from pathlib import Path
from PyPDF2 import PdfReader

def read_pdf(file_path):
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        reader = PdfReader(str(p))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(pages)
    except Exception as e:  # Catch generic exceptions
        raise ValueError(f"Failed to read PDF file: {file_path}. Error: {e}")