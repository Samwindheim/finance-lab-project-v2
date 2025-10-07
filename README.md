# PDF RAG Pipeline

A modular RAG (Retrieval-Augmented Generation) pipeline for extracting financial data from PDF documents.

## Setup

1. **Create virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API key:**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

The `run.sh` script provides a simple CLI for all operations:

### Index a PDF
```bash
run.sh index pdfs/your_document.pdf
```

### Query the database
```bash
run.sh query "Find the underwriter and guarantor section"
```

### Interactive query mode
```bash
run.sh interactive
```
Type queries and get instant results. Type `quit` to exit.

### Show index information
```bash
run.sh info
```

### Clear the vector database
```bash
run.sh clear
```

### Options
- `--index PATH`: Specify a custom index path (default: `faiss_index`)
- `-k N` or `--top-k N`: Number of results to return (default: 3)
- `-y` or `--yes`: Skip confirmation when clearing (for `clear` command)

## Examples

```bash
# Index your PDF
run.sh index pdfs/BAT_2025‑08‑25_Informationsdokument.pdf

# Query for specific information
run.sh query "risk factors" -k 5

# Use interactive mode for multiple queries
run.sh interactive

# Check what's indexed
run.sh info

# Clear everything and start fresh
run.sh clear -y
```

## Architecture

- **`pdf_indexer.py`**: Core indexer class with PDF extraction, embedding generation, and FAISS storage
- **`run.py`**: CLI interface for easy interaction
- **`run.sh`**: Run script to make things easy
- **FAISS**: Vector database for fast similarity search
- **OpenAI Embeddings**: text-embedding-3-small model for semantic understanding

## Features

✅ Page-by-page PDF text extraction  
✅ Semantic search with OpenAI embeddings  
✅ Persistent vector storage with FAISS  
✅ Metadata tracking (document ID, page number)  
✅ Clean CLI interface  
✅ Interactive query mode  

## Next Steps (Step 2)

- Vision-based extraction for tables and images
- Multi-modal embeddings
- Table structure parsing
