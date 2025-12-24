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

def load_json_file(file_path: str) -> list:
    """Loads a JSON file and returns its content."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
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
        print("\nError: Failed to decode JSON from model response.")
        print("Raw response:")
        print(json_string)
        return None

def find_issue_id(source_path: str, pdf_sources: list, html_sources: list) -> str | None:
    """Finds the issue_id for a given source path from all available source files."""
    # Check for exact matches first (highest priority)
    
    # HTML/URL check
    for source in html_sources:
        if source.get("source_url") == source_path:
            return source.get("issue_id") or source.get("warrant_id") or source.get("convertible_id")
    
    # PDF check
    pdf_filename = os.path.basename(source_path)
    for source in pdf_sources:
        if source.get("source_url") == pdf_filename:
            return source.get("issue_id")

    # If no exact match, try partial matches
    for source in html_sources:
        if source_path in (source.get("source_url") or ""):
            return source.get("issue_id") or source.get("warrant_id") or source.get("convertible_id")
            
    for source in pdf_sources:
        if source_path in (source.get("source_url") or ""):
            return source.get("issue_id")
            
    return None
