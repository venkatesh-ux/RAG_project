import pytest
from src.pdf_processor import read_pdf

def test_read_pdf():
    # Test with a valid PDF file
    pdf_path = 'data/books/sample.pdf'
    content = read_pdf(pdf_path)
    assert isinstance(content, str), "Content should be a string"
    assert len(content) > 0, "Content should not be empty"

def test_read_invalid_pdf():
    # Test with an invalid PDF file path
    invalid_pdf_path = 'data/books/invalid.pdf'
    with pytest.raises(FileNotFoundError):
        read_pdf(invalid_pdf_path)