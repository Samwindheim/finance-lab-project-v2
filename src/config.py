"""
Central Configuration File.

This file contains all the core settings that control the behavior of the data extraction
pipeline. It defines embedding models, API endpoints, file paths, and other constants
that are used throughout the application.

Separating configuration into its own file allows for easy updates and management of
settings without modifying the application's core logic.
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
# or models/gemini-1.5-flash
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

# --- Source File Paths ---
PDF_SOURCES_FILE = BASE_DIR / "source_files" / "pdf-sources.json"
HTML_SOURCES_FILE = BASE_DIR / "source_files" / "html-sources.json"

# --- Extraction Queries ---
EXTRACTION_QUERIES = {
    "investors": "Section or table listing investor names, subscription commitments (teckningsåtaganden or teckningsförbindelser), guarantee commitments (garantiåtaganden or garanti)",
    "temp": "future query text"
}

#Old under writere query: "Tabell över teckningsförbindelser, teckningsåtaganden och garantiåtaganden med kolumner för namn, belopp i SEK, och andel i procent (%)",

# Section or table listing investor names, subscription commitments (teckningsåtaganden or teckningsförbindelser), guarantee commitments (garantiåtaganden or garanti) or teckning av aktier

# garantiåtaganden or  garanti and teckningsåtaganden or teckningsförbindelser

# teckningsförbindelse och garantiåtagande