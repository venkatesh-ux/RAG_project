def read_pdf(file_path):
    from PyPDF2 import PdfReader

    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_pdf(pdf_file):
    try:
        text = read_pdf(pdf_file)
        return text
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")
        return None

def process_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    if text:
        # Further processing can be added here (e.g., text cleaning, splitting)
        return text
    return None