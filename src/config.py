# --- Configuration file for the PDF RAG pipeline ---

# PDF Indexer settings
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 1536
CHUNK_SIZE = 700  # Target size of each text chunk in tokens
CHUNK_OVERLAP = 100  # Number of tokens to overlap between chunks
EMBEDDING_BATCH_SIZE = 100 # Number of chunks to process in a single API call

# Vision settings
GEMINI_MODEL = "models/gemini-2.5-flash"
PROMPT_FILE = "prompt.txt"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Directory settings
from pathlib import Path

# --- Directories ---
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "pdfs"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
OUTPUT_IMAGE_DIR = BASE_DIR / "output_images"
OUTPUT_JSON_DIR = BASE_DIR / "output_json"
PROMPTS_DIR = BASE_DIR / "prompts"

# --- Extraction Queries ---
EXTRACTION_QUERIES = {
    "underwriters": "Tabell över teckningsförbindelser, teckningsåtaganden och garantiåtaganden med kolumner för namn, belopp i SEK, och andel i procent (%)",
    "handle": "future query text"
}
