import pytest
from src.retriever import Retriever

def test_retriever_initialization():
    retriever = Retriever()
    assert retriever is not None

def test_retrieve_documents():
    retriever = Retriever()
    question = "What is the main topic of the document?"
    results = retriever.retrieve(question)
    assert isinstance(results, list)
    assert len(results) > 0

def test_retrieve_with_no_results():
    retriever = Retriever()
    question = "This question should return no results."
    results = retriever.retrieve(question)
    assert results == []  # Expecting an empty list for no results

def test_retrieve_documents_with_context():
    retriever = Retriever()
    question = "Explain the concept of overfitting."
    results = retriever.retrieve(question)
    assert isinstance(results, list)
    assert len(results) > 0
    assert all("overfitting" in doc.page_content for doc in results)  # Check if results contain relevant content