# PDF RAG Pipeline

A modular RAG (Retrieval-Augmented Generation) pipeline for extracting financial data from PDF documents using semantic search and vision models.

## Setup

1.  **Create virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Set up API keys:**
    Create a `.env` file in the project root:
    ```
    GEMINI_API_KEY=your_gemini_api_key_here
    OPENAI_API_KEY=your_openai_api_key_here
    ```  

## Usage

The `run.sh` script provides a simple CLI for all operations:

### Index a PDF
Indexes all pages of a PDF document for semantic search.
```bash
./run.sh index pdfs/your_document.pdf
```

### Query the database
Performs a semantic search over the indexed documents.
```bash
./run.sh query pdfs/your_document.pdf "query text" -n 5
```
*   The `-n` flag specifies the number of top results to return. (Default is 3)

### Extract Underwriter Data
Automatically finds the most relevant page for underwriter data, extracts it as an image, and uses a vision model to return structured JSON.
```bash
./run.sh extract_underwriters pdfs/your_document.pdf
```
*   The output is saved to a file in the `output_json/` directory.

### Clear the vector database
Deletes the existing index for a specific pdf.
```bash
./run.sh clear pdfs/your_document.pdf
```

## Architecture

-   **`pdf_indexer.py`**: Core class for PDF parsing, text embedding, and FAISS indexing.
-   **`cli.py`**: Command-line interface for all user interactions.
-   **`vision.py`**: Module for vision-based data extraction using the Gemini API.
-   **`config.py`**: Configuration file.
-   **FAISS**: Vector database for efficient similarity search.
-   **OpenAI Embeddings**: `text-embedding-3-large` for semantic text representation.
-   **Gemini 2.5 Flash**: Vision model for structured data extraction from images.

## Features

✅ Page-by-page PDF text extraction  
✅ Semantic search with OpenAI embeddings  
✅ Vision-based data extraction with Gemini  
✅ Persistent vector storage with FAISS  
✅ Clean, simplified CLI for core operations
