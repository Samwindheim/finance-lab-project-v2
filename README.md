# Finance Lab: Financial Data Extraction Pipeline

A RAG pipeline for extracting structured financial data from PDF documents and HTML pages using semantic search and vision models. Designed for Swedish financial documents (prospectuses, memorandums, press releases) related to rights issues, IPOs, and directed offerings.

## Features

- **Dual Extraction Modes**: Context-aware (Historical) and Zero-context (New).
- **Multi-format support**: PDF (Gemini Vision) and HTML (Gemini Text).
- **Semantic Search**: FAISS vector indexing for precise page selection.
- **Database Staging**: Saves extractions to a staging table (`ai_extractions`) for admin review.
- **Smart Filtering**: Automatically selects relevant fields based on document and issue types.

## Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure `.env` file:**
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   DATABASE_URL=mysql+pymysql://user:password@host:port/database
   ```

## Usage

### 1. Historical Mode (Context-Aware)
Use this for documents already registered in the database. It uses existing metadata for smart filtering.
```bash
# Extract from document URL or ID
./run.sh extract-historical "https://example.com/prospectus.pdf"

# Extract for entire issue
./run.sh extract-historical --issue-id "50898375-798c-4242-bcde-1aeddd914a55"

# Extract specific field
./run.sh extract-historical --issue-id "..." --extraction-field investors
```

### 2. New Mode (Zero-Context)
Use this for entirely new documents not yet in the database.
```bash
./run.sh extract-new "https://example.com/new_doc.pdf" --extraction-field important_dates
```

**Extraction Fields:** `investors`, `important_dates`, `offering_terms`, `offering_outcome`, `general_info`

### 3. Evaluate Accuracy
Compare AI results against ground truth in the database.
```bash
./run.sh test --issue-id "50898375-798c-4242-bcde-1aeddd914a55"
./run.sh batch_test -n 10 --exclude-fields general_info
```

### 4. Developer Utilities
```bash
./run.sh index pdfs/document.pdf              # Index PDF
./run.sh query pdfs/document.pdf "query" -n 5  # Semantic search
./run.sh clear pdfs/document.pdf -y             # Clear index
./run.sh html-text "https://example.com"       # Extract HTML text
```

## How It Works

1. **Identification**: Resolves documents against the database `sources` table or treats as "New".
2. **Field Filtering**: Selects fields from `extraction_definitions.json` based on document/issue context.
3. **RAG Extraction**:
   - **PDFs**: Semantic search → page selection → image + text → Gemini vision model.
   - **HTML**: Full text extraction → Gemini text model.
4. **Validation**: Validates output against Pydantic models in `src/models.py`.
5. **Storage**: 
   - Saves JSON results to `output_json/`.
   - **UPSERTs** data into the `ai_extractions` staging table (keyed by `source_url` and `extraction_field`).

## Project Structure

```
.
├── src/                  # Core logic
│   ├── main.py          # CLI entry point
│   ├── extraction_logic.py
│   ├── models.py        # Pydantic models
│   ├── pdf_indexer.py   # FAISS indexing
│   ├── database.py      # SQL connection & Staging
│   └── extraction_definitions.json
├── prompts/             # Field-specific prompts
├── output_json/         # Extraction results
├── tests/               # Evaluation tools
└── run.sh               # CLI wrapper
```

## Notes

- **Uniqueness**: The database staging area guarantees one entry per `(source_url, extraction_field)` pair.
- **Source Pages**: Extractions include `source_pages` at the field level for easy verification.
- **Accuracy**: Prioritizes explicit data points; prompts forbid LLM-side calculations.
