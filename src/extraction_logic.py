"""
Core Data Extraction and Processing Logic.

This module contains the primary functions for performing the Retrieval-Augmented Generation (RAG)
extraction from various document types. It encapsulates the logic for querying the vector index,
selecting relevant pages, calling the vision/language models, and post-processing the results.

These functions are designed to be reusable and are orchestrated by higher-level scripts
like `run_extraction.py`.
"""
import os
import json
from typing import List, Dict

from pdf_indexer import PDFIndexer
from vision import get_json_from_image, get_json_from_text
from html_processor import extract_text_from_html
import config
from utils import find_issue_id, clean_and_parse_json

# --- Load sources data once for performance ---
# This data is needed for `find_issue_id` in the post-processing step.
SOURCES_DATA = None
HTML_SOURCES_DATA = None

def _load_sources_data():
    """Lazy loader for the sources data to avoid loading it if not needed."""
    global SOURCES_DATA, HTML_SOURCES_DATA
    if SOURCES_DATA is None:
        from utils import load_json_file
        SOURCES_DATA = load_json_file(config.PDF_SOURCES_FILE)
    if HTML_SOURCES_DATA is None:
        from utils import load_json_file
        HTML_SOURCES_DATA = load_json_file(config.HTML_SOURCES_FILE)


def select_consecutive_pages(results: List[Dict]) -> List[int]:
    """
    Selects a consecutive block of pages around the top search results from a PDF.
    This logic is crucial for capturing tables that span multiple pages.
    """
    if not results:
        return []

    SIMILARITY_DISTANCE_THRESHOLD = 1.05 # 5% threshold

    # Start with the top result and include the second result if its score is close.
    top_result = results[0]
    pages_to_extract = [top_result]
    
    # Check if second result is close enough to include
    if len(results) > 1:
        second_result = results[1]
        # If the second result's distance is within the threshold, include its page.
        if second_result['distance'] <= top_result['distance'] * SIMILARITY_DISTANCE_THRESHOLD:
            if second_result['page_number'] != top_result['page_number']:
                pages_to_extract.append(second_result)

    results_by_page = {res['page_number']: res for res in results}
    
    # --- Expand page selection around the top result ---
    # Look forward to find consecutive pages.
    current_page = top_result['page_number'] + 1
    # Always include the page immediately following the top result for context.
    if current_page in results_by_page:
        pages_to_extract.append(results_by_page[current_page])
    else:
        # Synthetic page to ensure the next page is included even if not in top results
        pages_to_extract.append({'page_number': current_page, 'distance': float('inf'), 'text': ''})
    
    current_page += 1
    while current_page in results_by_page:
        pages_to_extract.append(results_by_page[current_page])
        current_page += 1

    # Look backward to find consecutive pages that are also in the search results.
    current_page = top_result['page_number'] - 1
    while current_page > 0 and current_page in results_by_page:
        pages_to_extract.append(results_by_page[current_page])
        current_page -= 1

    # Remove duplicates by page number, keeping the first occurrence
    unique_pages = []
    seen_page_numbers = set()
    for page in pages_to_extract:
        if page['page_number'] not in seen_page_numbers:
            unique_pages.append(page)
            seen_page_numbers.add(page['page_number'])
    
    # Sort the collected pages by page number to ensure correct order
    unique_pages.sort(key=lambda p: p['page_number'])

    return [p['page_number'] for p in unique_pages]


def post_process_and_save(parsed_json: dict, source_path: str, extraction_field: str, source_pages: list, output_path: str):
    """Post-processes and saves the extracted JSON data to a specified file path."""
    _load_sources_data() # Ensure source data is loaded
    
    issue_id = find_issue_id(source_path, SOURCES_DATA, HTML_SOURCES_DATA)
    if not issue_id:
        print(f"Warning: Could not find issue_id for '{os.path.basename(source_path)}'.")

    # The parsed_json from the model should contain the extraction_field as a key
    final_output = {
        "issue_id": issue_id,
        "source_document": os.path.basename(source_path),
        "source_pages": source_pages,
        extraction_field: parsed_json.get(extraction_field, []) # Default to empty list
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"  - Successfully extracted and saved data to: {output_path}")
    return output_path


def extract_from_pdf(pdf_path: str, search_query: str, extraction_prompt: str, extraction_field: str, output_path: str, page_selection_strategy: str = "consecutive"):
    """
    Reusable logic to perform a full RAG extraction on a single PDF document.
    """
    print(f"\n--- Processing PDF: {os.path.basename(pdf_path)} ---")
    
    # Simplified index path generation
    index_name = os.path.splitext(os.path.basename(pdf_path))[0]
    index_path = os.path.join(config.FAISS_INDEX_DIR, index_name)

    indexer = PDFIndexer(index_path=index_path)
    if indexer.index.ntotal == 0:
        print(f"  - Index not found for {os.path.basename(pdf_path)}. Building it now...")
        indexer.index_pdf(pdf_path)

    # 1. Query the index
    print(f"  - Querying for relevant pages (strategy: {page_selection_strategy})...")
    results = indexer.query(search_query, top_k=5)

    if not results:
        print("  - Could not find any relevant pages.")
        return None

    # 2. Select pages based on the specified strategy
    page_numbers = []
    if page_selection_strategy == "top_hit":
        # For simple fields, just take the page from the single best result.
        page_numbers = [results[0]['page_number']]
    else: # Default to "consecutive"
        page_numbers = select_consecutive_pages(results)
        
    print(f"  - Selected {len(page_numbers)} pages to analyze: {page_numbers}")

    # 3. Extract images and text for context
    saved_image_paths = []
    page_texts = []
    for page_num in page_numbers:
        # Check if the page exists in the document before trying to extract
        if page_num > 0 and page_num <= indexer.get_page_count(pdf_path):
            img_path = indexer.extract_page_as_image(pdf_path, page_num)
            if img_path:
                saved_image_paths.append(img_path)
            
            text = indexer.get_text_for_page(pdf_path, page_num)
            if text:
                page_texts.append(text)
    
    if not saved_image_paths:
        print("  - Error: Could not extract any images for the identified pages.")
        return None

    # 4. Call LLM and save result
    combined_text = "\\n\\n--- Page Separator ---\\n\\n".join(page_texts)
    json_data = get_json_from_image(saved_image_paths, combined_text, prompt_text=extraction_prompt, extraction_type=extraction_field)
    if json_data:
        parsed_json = clean_and_parse_json(json_data)
        if parsed_json:
            return post_process_and_save(parsed_json, pdf_path, extraction_field, page_numbers, output_path)
    return None


def extract_from_html(html_path: str, extraction_prompt: str, extraction_field: str, output_path: str):
    """
    Reusable logic to perform a full-text extraction on a single HTML document.
    """
    print(f"--- Processing HTML: {os.path.basename(html_path)} ---")
    
    # 1. Extract text
    text = extract_text_from_html(html_path, preserve_tables=True)
    if not text:
        print("  - Could not extract any text from the HTML file.")
        return None

    # 2. Call LLM and save result
    json_data = get_json_from_text(text, prompt_text=extraction_prompt, extraction_type=extraction_field)
    if json_data:
        parsed_json = clean_and_parse_json(json_data)
        if parsed_json:
            # For HTML, we consider it as a single "page"
            return post_process_and_save(parsed_json, html_path, extraction_field, [1], output_path)
    return None


def merge_and_finalize_outputs(issue_id: str, extraction_field: str, temp_files: List[str], final_output_path: str):
    """
    Merges data from multiple temporary files, deduplicates based on the extraction
    field type, and saves to a final output file.
    """
    if not temp_files:
        print("  - No temporary files to merge.")
        return

    combined_data = []
    source_docs_processed = []

    for file_path in temp_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = data.get(extraction_field)
                
                # Skip if the key exists but has no value (null, empty list)
                if not items:
                    continue

                # The 'investors' field is a list that should be extended.
                # Other fields (like 'record_date') are single values to be appended.
                if isinstance(items, list):
                    combined_data.extend(items)
                else:
                    combined_data.append(items)
                
                source_docs_processed.append(data.get("source_document", os.path.basename(file_path)))

        except (json.JSONDecodeError, FileNotFoundError):
            print(f"  - Warning: Could not read or parse temp file {os.path.basename(file_path)}. Skipping.")
            continue

    # --- Finalize based on the data type ---
    final_field_value = None
    
    if extraction_field == "investors":
        # Handle deduplication for a list of investor dicts
        unique_items = []
        seen_entries = set()
        for item in combined_data:
            # Ensure item is a dictionary before using .get()
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            level = item.get("level")
            unique_key = (name, level)
            
            if name and unique_key not in seen_entries:
                unique_items.append(item)
                seen_entries.add(unique_key)
        
        # --- Apply final formatting to numbers ---
        formatted_items = []
        for investor in unique_items:
            # Format amount_in_cash to string with 3 decimal places
            amount = investor.get("amount_in_cash")
            if isinstance(amount, (int, float)):
                investor["amount_in_cash"] = f"{amount:.3f}"
            
            # Format amount_in_percentage to string
            percent = investor.get("amount_in_percentage")
            if isinstance(percent, (int, float)):
                investor["amount_in_percentage"] = str(percent)
            
            formatted_items.append(investor)

        final_field_value = formatted_items
    else:
        # For simple fields (like record_date), find unique values and take the first one.
        # Using a set handles deduplication automatically and ignores None.
        unique_values = sorted(list(set(item for item in combined_data if item is not None)))
        if len(unique_values) > 1:
            print(f"  - Warning: Multiple different values found for '{extraction_field}': {unique_values}. Using the first one.")
        if unique_values:
            final_field_value = unique_values[0]

    # Final combined output structure
    final_output = {
        "issue_id": issue_id,
        extraction_field: final_field_value,
        "contributing_sources": sorted(list(set(source_docs_processed)))
    }

    # Save the combined file
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    item_count = len(final_field_value) if isinstance(final_field_value, list) else 1 if final_field_value else 0
    print(f"\n  - Successfully merged {item_count} unique item(s) from {len(source_docs_processed)} source(s).")
    print(f"  - Combined data saved to: {final_output_path}")
