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
from logger import setup_logger

logger = setup_logger(__name__)

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

def find_sources_by_issue_id(issue_id: str, pdf_sources: list, html_sources: list) -> tuple[list, list]:
    """Finds all sources matching a given issue_id."""
    pdf_matches = [s for s in pdf_sources if s.get("issue_id") == issue_id]
    html_matches = [s for s in html_sources if s.get("issue_id") == issue_id]
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

def find_document_info(source_path: str, pdf_sources: list, html_sources: list, issue_id: str = None) -> dict:
    """
    Finds document-related information (issue_id, id, etc.) for a given source path.
    If issue_id is provided, prioritizes matches for that specific issue when multiple sources share the same URL.
    Returns a dictionary with the info.
    """
    info = {"issue_id": None, "doc_id": None}
    
    # Helper function to check if source matches the issue_id filter
    def matches_issue(source):
        if not issue_id:
            return True
        source_issue_id = source.get("issue_id") or source.get("warrant_id") or source.get("convertible_id")
        return source_issue_id == issue_id
    
    # Check for document ID matches first (if source_path is already an ID)
    for source in html_sources + pdf_sources:
        if source.get("id") == source_path:
            if matches_issue(source):
                info["doc_id"] = source.get("id")
                info["issue_id"] = source.get("issue_id") or source.get("warrant_id") or source.get("convertible_id")
                return info

    # HTML/URL check
    for source in html_sources:
        if source.get("source_url") == source_path:
            if matches_issue(source):
                info["doc_id"] = source.get("id")
                info["issue_id"] = source.get("issue_id") or source.get("warrant_id") or source.get("convertible_id")
                return info
    
    # PDF check
    pdf_filename = os.path.basename(source_path)
    for source in pdf_sources:
        if source.get("source_url") == pdf_filename:
            if matches_issue(source):
                info["doc_id"] = source.get("id")
                info["issue_id"] = source.get("issue_id")
                return info

    # Partial matches
    for source in html_sources:
        if source_path in (source.get("source_url") or ""):
            if matches_issue(source):
                info["doc_id"] = source.get("id")
                info["issue_id"] = source.get("issue_id") or source.get("warrant_id") or source.get("convertible_id")
                return info
            
    for source in pdf_sources:
        if source_path in (source.get("source_url") or ""):
            if matches_issue(source):
                info["doc_id"] = source.get("id")
                info["issue_id"] = source.get("issue_id")
                return info
    
    # If we didn't find a match with the issue_id filter, try without it as fallback
    if issue_id:
        logger.warning(f"No document found for '{source_path}' with issue_id '{issue_id}'. Trying without filter...")
        return find_document_info(source_path, pdf_sources, html_sources, issue_id=None)
            
    return info

def find_issue_id(source_path: str, pdf_sources: list, html_sources: list) -> str | None:
    """Finds the issue_id for a given source path or document ID."""
    info = find_document_info(source_path, pdf_sources, html_sources)
    return info["issue_id"]

def find_source_by_doc_id(doc_id: str, pdf_sources: list, html_sources: list) -> dict | None:
    """Finds a specific source document entry by its unique ID."""
    for source in html_sources + pdf_sources:
        if source.get("id") == doc_id:
            return source
    return None

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
