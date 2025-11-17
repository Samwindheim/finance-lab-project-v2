# PDF RAG Pipeline

A modular RAG (Retrieval-Augmented Generation) pipeline for extracting financial data from PDF documents using semantic search and vision models.

## Project Structure

```
.
├── faiss_index/      # Stores persistent FAISS vector indexes
├── output_images/    # Stores temporary images for vision model analysis
├── output_json/      # Stores the final structured JSON output
├── pdfs/             # Directory for input PDF documents
├── prompts/          # Contains prompts for the vision model
├── src/              # Main source code for the pipeline
├── tests/            # Test scripts and data
├── .env              # API keys (not committed)
├── config.py         # Main configuration file
└── run.sh            # Main execution script
```

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
Indexes all pages of a PDF document for semantic search. Each PDF gets its own vector index in the `faiss_index/` directory.
```bash
./run.sh index pdfs/your_document.pdf
```

### Query the database
Performs a semantic search over an indexed PDF.
```bash
./run.sh query pdfs/your_document.pdf "your query text" -n 5
```
*   The `-n` flag specifies the number of top results to return. (Default is 3)

### Extract Structured Data
Extracts specific data types from a PDF using a multi-step RAG process.
```bash
./run.sh extract underwriters pdfs/your_document.pdf
```
*   The first argument is the `extraction_type`, which must be defined in `config.py` and have a corresponding prompt in the `prompts/` directory.
*   The output is saved to a file in the `output_json/` directory.

### Clear the vector database
Deletes the existing FAISS index for a specific PDF.
```bash
./run.sh clear pdfs/your_document.pdf
```

## Architecture

-   **`src/cli.py`**: Command-line interface for all user interactions, powered by `argparse`.
-   **`src/pdf_indexer.py`**: Core class for PDF parsing (`PyMuPDF`), text chunking (`TikToken`), embedding generation (`OpenAI`), and vector indexing (`FAISS`).
-   **`src/vision.py`**: Module for vision-based data extraction using the Gemini API. It constructs the multi-modal payload (prompt, text, images) and handles the API request.
-   **`src/config.py`**: Central configuration file for models, prompts, and directory paths.
-   **FAISS**: Vector database for efficient similarity search on text embeddings.
-   **OpenAI Embeddings**: `text-embedding-3-large` for semantic text representation.
-   **Gemini 2.5 Flash**: Vision model for structured data extraction from images.

## Features

✅ Page-by-page PDF text extraction
✅ Semantic search with OpenAI embeddings
✅ Vision-based data extraction with Gemini
✅ Persistent vector storage with FAISS
✅ Clean, simplified CLI for core operations
✅ Highly modular and configurable

