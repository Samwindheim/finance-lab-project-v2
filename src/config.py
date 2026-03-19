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
import os
from pathlib import Path

# --- Directories ---
BASE_DIR = Path(__file__).resolve().parent.parent

# Detect if running in AWS Lambda
IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None

if IS_LAMBDA:
    # Lambda can only write to /tmp
    PDF_DIR = Path("/tmp/pdfs")
    FAISS_INDEX_DIR = Path("/tmp/faiss_index")
    OUTPUT_IMAGE_DIR = Path("/tmp/output_images")
    OUTPUT_JSON_DIR = Path("/tmp/output_json")
else:
    PDF_DIR = BASE_DIR / "pdfs"
    FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
    OUTPUT_IMAGE_DIR = BASE_DIR / "output_images"
    OUTPUT_JSON_DIR = BASE_DIR / "output_json"

PROMPTS_DIR = BASE_DIR / "prompts"

# Ensure directories exist (only needed for local, Lambda creates /tmp subdirs on the fly or we do it in code)
for d in [PDF_DIR, FAISS_INDEX_DIR, OUTPUT_IMAGE_DIR, OUTPUT_JSON_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Logging settings ---
LOG_LEVEL = "INFO" # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "  %(levelname)-7s | %(message)s"
