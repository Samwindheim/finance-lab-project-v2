"""
Main Orchestrator for the Definition-Driven Extraction Pipeline.

This script serves as the primary entry point for the data extraction process. 
It can be run on an entire issue or a specific document link.

Usage Examples:
1. Run on a specific document:
   python src/run_extraction.py <document_link> --extraction-field investors

2. Run on all documents for an issue:
   python src/run_extraction.py --issue-id <issue_id>

3. Run a specific field for an entire issue:
   python src/run_extraction.py --issue-id <issue_id> --extraction-field investors
"""
import argparse
import json
import os
import sys
import shutil

# Add project root to the Python path to allow root-level imports like 'config'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_json_file, find_sources_by_issue_id, find_issue_id
import config
from extraction_logic import extract_from_pdf, extract_from_html, merge_and_finalize_outputs

def load_extraction_definitions():
    """Loads the extraction definitions from the JSON file."""
    definitions_path = os.path.join(os.path.dirname(__file__), 'extraction_definitions.json')
    definitions = load_json_file(definitions_path)
    if not definitions:
        print(f"Error: Could not load extraction definitions from {definitions_path}")
        sys.exit(1)
    return definitions

def run_single_extraction(issue_id: str, extraction_field: str, definitions: dict, pdf_matches: list, html_matches: list):
    """Runs the full extraction and merge pipeline for a single field."""
    field_definition = definitions.get(extraction_field)
    if not field_definition:
        print(f"Warning: No definition found for '{extraction_field}'. Skipping.")
        return

    # --- Step 1: Load the Prompt Text from its File ---
    prompt_filename = f"{extraction_field}.txt"
    prompt_path = os.path.join(config.PROMPTS_DIR, prompt_filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            extraction_prompt = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found for '{extraction_field}'. Please create '{prompt_filename}' in the prompts directory.")
        return

    # --- Step 2: Get Definition Details ---
    source_types = field_definition.get("source_types", [])
    semantic_search_query = field_definition.get("semantic_search_query")
    page_selection_strategy = field_definition.get("page_selection_strategy", "consecutive")
    
    if not all([source_types, semantic_search_query]):
         print(f"Error: Definition for '{extraction_field}' is missing required keys (source_types, semantic_search_query). Skipping.")
         return

    print(f"  - Loaded definition. Target source types: {source_types}")

    # --- Step 3: Source-Specific Extraction ---
    # Process source types in order, stopping as soon as data is found
    temp_output_files = []
    temp_dir = os.path.join(config.OUTPUT_JSON_DIR, 'temp', f"{issue_id}_{extraction_field}")
    os.makedirs(temp_dir, exist_ok=True)
    
    data_found = False  # Track if we've successfully extracted data

    for doc_type in source_types:
        if data_found:
            print(f"  - Data already extracted. Skipping remaining source types.")
            break
        
        print(f"  - Looking for '{doc_type}' documents...")
            
        # Check PDFs with this source_type
        for pdf_info in pdf_matches:
            if pdf_info.get("source_type") == doc_type:
                pdf_filename = pdf_info.get("source_url")
                if not pdf_filename: continue
                pdf_path = os.path.join(config.PDF_DIR, pdf_filename)
                
                temp_output_filename = f"{os.path.splitext(pdf_filename)[0]}_{extraction_field}.json"
                temp_output_path = os.path.join(temp_dir, temp_output_filename)

                result_path = extract_from_pdf(
                    pdf_path=pdf_path,
                    search_query=semantic_search_query,
                    extraction_prompt=extraction_prompt,
                    extraction_field=extraction_field,
                    output_path=temp_output_path,
                    page_selection_strategy=page_selection_strategy
                )
                if result_path:
                    temp_output_files.append(result_path)
                    data_found = True
                    break  # Stop processing more documents once we have data
        
        if data_found:
            break  # Exit outer loop if we found data in PDFs
        
        # Check HTMLs with this source_type
        for html_info in html_matches:
            if html_info.get("source_type") == doc_type:
                html_url = html_info.get("source_url")
                if not html_url: continue
                
                safe_filename = "".join(c for c in os.path.basename(html_url) if c.isalnum() or c in ('-', '_')).rstrip()
                temp_output_filename = f"{safe_filename}_{extraction_field}.json"
                temp_output_path = os.path.join(temp_dir, temp_output_filename)

                result_path = extract_from_html(
                    html_path=html_url,
                    extraction_prompt=extraction_prompt,
                    extraction_field=extraction_field,
                    output_path=temp_output_path
                )
                if result_path:
                    temp_output_files.append(result_path)
                    data_found = True
                    break  # Stop processing more documents once we have data

    # --- Step 4: Merging and Finalization ---
    print(f"\n  - Merging {len(temp_output_files)} extraction output(s)...")
    
    if temp_output_files:
        final_filename = f"{issue_id}_extraction.json"
        final_output_path = os.path.join(config.OUTPUT_JSON_DIR, final_filename)
        
        merge_and_finalize_outputs(
            issue_id=issue_id,
            extraction_field=extraction_field,
            temp_files=temp_output_files,
            final_output_path=final_output_path
        )
        
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    else:
        print("  - No results to merge.")

def main():
    parser = argparse.ArgumentParser(description="Run extractions for a given issue ID or document.")
    parser.add_argument('source_link', nargs='?', help='A specific document filename or URL to process.')
    parser.add_argument('--issue-id', help='The issue ID to process. (Auto-detected if source_link is provided)')
    parser.add_argument('--extraction-field', help='A specific field to extract. If omitted, all fields will be extracted.')
    args = parser.parse_args()

    source_link = args.source_link
    issue_id = args.issue_id
    extraction_field = args.extraction_field

    # --- Step 1: Identify Issue ID and Target Document ---
    pdf_sources_data = load_json_file(config.PDF_SOURCES_FILE)
    html_sources_data = load_json_file(config.HTML_SOURCES_FILE)

    if source_link:
        detected_id = find_issue_id(source_link, pdf_sources_data, html_sources_data)
        if not detected_id and not issue_id:
            print(f"Error: Could not find an issue_id associated with '{source_link}'. Please provide --issue-id manually.")
            sys.exit(1)
        
        # Prefer provided issue_id, fall back to detected
        issue_id = issue_id or detected_id
        target_document = source_link
        print(f"--- Document mode: '{source_link}' | Issue ID: '{issue_id}' ---")
    elif issue_id:
        target_document = None
        print(f"--- Issue mode: '{issue_id}' ---")
    else:
        print("Error: Please provide either a document link (positional) or an --issue-id.")
        parser.print_help()
        sys.exit(1)

    definitions = load_extraction_definitions()

    # --- Step 2: Identify all fields to be processed ---
    fields_to_process = []
    if extraction_field:
        if extraction_field in definitions:
            fields_to_process.append(extraction_field)
            print(f"--- Processing field: '{extraction_field}' ---")
        else:
            print(f"Error: Extraction field '{extraction_field}' not found in definitions.")
            sys.exit(1)
    else:
        fields_to_process = list(definitions.keys())
        print(f"Found {len(fields_to_process)} fields to extract: {', '.join(fields_to_process)}")

    # --- Step 3: Find all source documents for the issue ---
    print("\n--- Identifying source documents ---")
    pdf_matches, html_matches = find_sources_by_issue_id(issue_id, pdf_sources_data, html_sources_data)

    # Filter matches if a target document is specified
    if target_document:
        pdf_matches = [m for m in pdf_matches if target_document in m.get("source_url", "")]
        html_matches = [m for m in html_matches if target_document in m.get("source_url", "")]
        print(f"  - Filtering for document: '{target_document}'")

    if not pdf_matches and not html_matches:
        print(f"Error: No source documents found for issue_id '{issue_id}'" + (f" and document '{target_document}'" if target_document else "") + ". Aborting.")
        sys.exit(1)
    
    print(f"Found {len(pdf_matches)} PDF(s) and {len(html_matches)} HTML document(s) for the issue.")

    # --- Step 4: Loop through and run each extraction ---
    for field in fields_to_process:
        print(f"\n{'='*60}")
        print(f"--- Processing field: '{field}' ---")
        run_single_extraction(issue_id, field, definitions, pdf_matches, html_matches)

    print(f"\n--- All processing for issue '{issue_id}' complete. ---")


if __name__ == "__main__":
    main()
