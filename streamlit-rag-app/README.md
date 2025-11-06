# Streamlit RAG Application

This project is a Streamlit application that allows users to ask questions and receive responses based on the content processed from a PDF file. The application utilizes machine learning techniques to extract information from the PDF and provide relevant answers to user queries.

## Project Structure

```
streamlit-rag-app
├── app.py                  # Main entry point for the Streamlit application
├── requirements.txt        # Dependencies required for the project
├── README.md               # Documentation for the project
├── .gitignore              # Files and directories to be ignored by Git
├── ML_Book.ipynb          # Jupyter notebook for processing the PDF and generating embeddings
├── src                     # Source code for the application
│   ├── config.py          # Configuration settings
│   ├── pdf_processor.py    # Functions for reading and extracting text from PDF files
│   ├── embeddings.py       # Handles the creation of embeddings from extracted text
│   ├── vectorstore.py      # Manages storage and retrieval of embeddings
│   ├── retriever.py        # Logic for retrieving relevant documents based on queries
│   └── ui.py              # Functions for building the Streamlit UI components
├── data                    # Directory for data files
│   └── books
│       └── sample.pdf      # Sample PDF file for testing
├── tests                   # Unit tests for the application
│   ├── test_pdf_processor.py # Tests for pdf_processor.py
│   └── test_retriever.py   # Tests for retriever.py
└── notebooks               # Additional notebooks for analysis
    └── ML_Book.ipynb      # Duplicate of the main Jupyter notebook
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd streamlit-rag-app
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

## Usage

- Open the application in your web browser.
- Upload a PDF file or use the provided sample PDF.
- Enter your question in the input field and submit.
- The application will process the PDF content and return relevant answers based on your query.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.