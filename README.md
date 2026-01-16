# Finance Lab: Financial Data Extraction Pipeline

A modular RAG (Retrieval-Augmented Generation) pipeline for extracting financial data from PDF documents using semantic search and vision models.

## Project Structure

```
.
├── faiss_index/          # Persistent FAISS vector indexes for PDFs
├── output_images/        # Temporary page images for LLM analysis
├── output_json/          # Final structured extraction results
├── pdfs/                 # Local directory for PDF sources
├── prompts/              # Field-specific extraction prompts (.txt)
├── reference_files/      # Source manifests (pdf-sources.json, etc.)
├── src/                  # Core application logic
│   ├── main.py           # Unified CLI entry point
│   ├── extraction_logic.py # Orchestrates RAG
│   ├── models.py         # Pydantic data models
│   ├── llm.py            # Gemini LLM interface
│   ├── pdf_indexer.py    # PDF indexing and semantic search
│   ├── html_processor.py # HTML text extraction
│   └── config.py         # Central configuration (models, paths, etc.)
├── run.sh                # CLI wrapper script
└── requirements.txt      # Python dependencies
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

Use `./run.sh` to access the unified CLI.

### 1. Extract Structured Data (Production)
Run the full pipeline on a document or an entire issue.
```bash
# Extract all relevant fields from a specific document
./run.sh extract "https://example.com/prospectus.pdf"
```
### 2. PDF Indexing (Developer Utilities)
Manual index management for PDFs.
```bash
# Index a PDF
./run.sh index pdfs/document.pdf

# Query the index (Semantic Search)
./run.sh query pdfs/document.pdf "teckningsåtaganden" -n 5

# Clear an index
./run.sh clear pdfs/document.pdf
```

## How It Works

1.  **Identification**: The system identifies the `issue_id` and `source_type` of the target document using manifests in `reference_files/`.
2.  **Smart Filtering**: It filters extraction fields from `extraction_definitions.json` that are relevant to that `source_type`.
3.  **RAG Process**: 
    - For PDFs, it performs semantic search to find the most relevant pages, extracts them as images and text, and sends them to Gemini.
    - For HTML, it extracts the full text and sends it to Gemini.
4.  **Validation**: Extracted JSON is validated against Pydantic models in `src/models.py`.
6.  **Merging**: The results are merged into `{issue_id}_extraction.json` in `output_json/`.
