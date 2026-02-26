"""
General Utility Functions.

This module provides a collection of helper functions that are used across various parts
of the data extraction pipeline. These functions handle common, reusable tasks such as
loading and parsing JSON files, finding data in source manifests, and cleaning up raw
output from language models.

Centralizing these functions here helps to reduce code duplication and improve maintainability.
"""

import json
import os
import requests
import pandas as pd
from typing import Optional, Tuple, List, Dict
from sqlalchemy import text
from logger import setup_logger

logger = setup_logger(__name__)

# Lazy-loaded DB instance to avoid circular imports or early connection
_db = None

def get_db():
    global _db
    if _db is None:
        from database import FinanceDB
        _db = FinanceDB()
    return _db

def load_json_file(file_path: str) -> list:
    """Loads a JSON file and returns its content."""
    if not os.path.exists(file_path):
        logger.error(f"File not found at {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON from {file_path}")
        return []

def find_sources_by_issue_id(issue_id: str) -> Tuple[List[Dict], List[Dict]]:
    """Finds all sources matching a given issue_id from the database."""
    db = get_db()
    df = db.find_sources_by_issue(issue_id)
    
    pdf_matches = df[df['source_url'].str.lower().str.endswith('.pdf', na=False)].to_dict('records')
    html_matches = df[~df['source_url'].str.lower().str.endswith('.pdf', na=False)].to_dict('records')
    
    return pdf_matches, html_matches

def clean_and_parse_json(json_string: str) -> dict | None:
    """Cleans a JSON string from LLM response and parses it."""
    if json_string.startswith("```json"):
        json_string = json_string[7:-3].strip()
    elif json_string.startswith("```"):
        json_string = json_string[3:-3].strip()

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from model response: {json_string}")
        return None

def find_document_info(source_path: str, issue_id: str = None) -> dict:
    """
    Finds document-related information (issue_id, id, etc.) for a given source path using the database.
    If issue_id is provided, prioritizes matches for that specific issue.
    """
    db = get_db()
    info = {"issue_id": None, "doc_id": None, "source_url": None}
    
    # 1. Try direct ID lookup
    source = db.find_source_by_id(source_path)
    if source:
        if not issue_id or source.get("issue_id") == issue_id:
            info["doc_id"] = source.get("id")
            info["issue_id"] = source.get("issue_id")
            info["source_url"] = source.get("source_url")
            return info

    # 2. Try direct URL lookup
    source = db.find_source_by_url(source_path)
    if source:
        if not issue_id or source.get("issue_id") == issue_id:
            info["doc_id"] = source.get("id")
            info["issue_id"] = source.get("issue_id")
            info["source_url"] = source.get("source_url")
            return info

    # 3. Try filename lookup (for PDFs)
    filename = os.path.basename(source_path)
    if filename.lower().endswith(".pdf"):
        # We might need a more flexible query for filenames if they aren't full URLs in DB
        query = text("SELECT * FROM sources WHERE source_url LIKE :filename")
        df = pd.read_sql(query, db.engine, params={"filename": f"%{filename}%"})
        if not df.empty:
            # Filter by issue_id if provided
            if issue_id:
                filtered = df[df['issue_id'] == issue_id]
                if not filtered.empty:
                    match = filtered.iloc[0]
                else:
                    match = df.iloc[0] # Fallback to first match if issue_id doesn't match
            else:
                match = df.iloc[0]
            
            info["doc_id"] = match.get("id")
            info["issue_id"] = match.get("issue_id")
            info["source_url"] = match.get("source_url")
            return info

    return info

def find_issue_id(source_path: str) -> str | None:
    """Finds the issue_id for a given source path or document ID using the database."""
    info = find_document_info(source_path)
    return info["issue_id"]

def find_source_by_doc_id(doc_id: str) -> dict | None:
    """Finds a specific source document entry by its unique ID from the database."""
    db = get_db()
    return db.find_source_by_id(doc_id)

def download_pdf(url: str, save_path: str) -> bool:
    """
    Downloads a PDF file from a URL and saves it to the specified path.
    Returns True if successful, False otherwise.
    """
    try:
        logger.info(f"Downloading PDF from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Successfully downloaded to: {save_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download PDF from {url}: {e}")
        return False
