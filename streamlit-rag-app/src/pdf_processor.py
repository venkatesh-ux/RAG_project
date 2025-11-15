from pathlib import Path
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

def read_pdf(file_path):
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        reader = PdfReader(str(p))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(pages)
    except PdfReadError as e:
        raise ValueError(f"Failed to read PDF file: {file_path}. The file may be corrupted or incomplete. Error: {e}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred while reading the PDF file: {file_path}. Error: {e}")