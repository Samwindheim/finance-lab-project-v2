"""
Central Configuration.

Defines embedding models, API endpoints, file paths, and logging constants
used throughout the application.
"""
# --- Configuration file for the PDF RAG pipeline ---

# PDF Indexer settings
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 1536 # Number of dimensions in the embedding vector
CHUNK_SIZE = 700  # Target size of each text chunk in tokens
CHUNK_OVERLAP = 100  # Number of tokens to overlap between chunks
EMBEDDING_BATCH_SIZE = 100 # Number of chunks to process in a single API call

# Vision settings
GEMINI_MODEL = "models/gemini-2.5-flash"
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

# --- Logging settings ---
LOG_LEVEL = "INFO" # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
