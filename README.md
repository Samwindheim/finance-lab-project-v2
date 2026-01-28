# Finance Lab: Financial Data Extraction Pipeline

A RAG pipeline for extracting structured financial data from PDF documents and HTML pages using semantic search and vision models. Designed for Swedish financial documents (prospectuses, memorandums, press releases) related to rights issues, IPOs, and directed offerings.

## Features

- Multi-format support (PDF and HTML)
- Semantic search with FAISS vector indexing
- Vision models (Google Gemini) for PDF extraction
- Smart field filtering based on document/issue type
- Database integration for accuracy evaluation
- Handles shared sources across multiple issues

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

### Extract Data
```bash
# Extract from document URL or ID
./run.sh extract "https://example.com/prospectus.pdf"
./run.sh extract "1420d1bf-9f8e-4f35-b140-76806484c99a"

# Extract for entire issue, processes each doc linked to that issue 1 at a time
./run.sh extract --issue-id "50898375-798c-4242-bcde-1aeddd914a55"

# Extract specific field
./run.sh extract --issue-id "..." --extraction-field investors
```

**Extraction Fields:** `investors`, `important_dates`, `offering_terms`, `offering_outcome`

### Evaluate Accuracy
```bash
./run.sh test --issue-id "50898375-798c-4242-bcde-1aeddd914a55"
```

### Developer Utilities
```bash
./run.sh index pdfs/document.pdf              # Index PDF
./run.sh query pdfs/document.pdf "query" -n 5  # Semantic search
./run.sh clear pdfs/document.pdf -y             # Clear index
./run.sh html-text "https://example.com"       # Extract HTML text
```

## How It Works

1. **Document Identification**: Identifies `issue_id` and `source_type` from manifests in `reference_files/`
2. **Field Filtering**: Selects relevant fields based on document/issue type from `extraction_definitions.json`
3. **RAG Extraction**:
   - PDFs: Semantic search → page selection → image + text → Gemini vision model
   - HTML: Full text extraction → Gemini text model
4. **Validation**: Validates against Pydantic models in `src/models.py`
5. **Merging**: Combines results from multiple documents into `{issue_id}_extraction.json`

## Project Structure

```
.
├── src/                  # Core logic
│   ├── main.py          # CLI entry point
│   ├── extraction_logic.py
│   ├── models.py        # Pydantic models
│   ├── pdf_indexer.py   # FAISS indexing
│   ├── database.py      # SQL connection
│   └── extraction_definitions.json
├── prompts/             # Field-specific prompts
├── reference_files/     # Source manifests (pdf/html-sources.json)
├── output_json/         # Extraction results
├── tests/               # Evaluation tools
└── run.sh               # CLI wrapper
```

## Configuration

- **Extraction Definitions** (`src/extraction_definitions.json`): Maps fields to source types and search queries
- **Prompts** (`prompts/*.txt`): Field-specific LLM instructions
- **Models** (`src/models.py`): Pydantic schemas for validation

## Output Format

Results saved to `output_json/{issue_id}_extraction.json`:
```json
{
  "document-name.pdf": {
    "issue_id": "...",
    "id": "...",
    "investors": [...],
    "important_dates": {...},
    "offering_terms": {...},
    "offering_outcome": {...}
  }
}
```

## Notes

- Prioritizes accuracy over completeness (missing data preferred over incorrect calculations)
- Prompts explicitly forbid calculations - only explicitly stated values extracted
- Database evaluation compares AI results against ground truth for accuracy measurement
