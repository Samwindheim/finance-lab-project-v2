# --- Configuration file for the PDF RAG pipeline ---

# PDF Indexer settings
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 1536

# Vision settings
GEMINI_MODEL = "models/gemini-2.5-flash"
PROMPT_FILE = "prompt.txt"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# CLI settings
UNDERWRITER_QUERY = "Tabell med teckningsåtaganden och garantiåtaganden, med namn, belopp, andel % och totalsumma i SEK"

# Directory settings
PDF_DIR = "pdfs"
INDEX_DIR = "faiss_index"
OUTPUT_IMAGE_DIR = "output_images"
OUTPUT_JSON_DIR = "output_json"
