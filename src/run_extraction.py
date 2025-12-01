import argparse
import json
import os
import sys
import shutil

# Add project root to the Python path to allow root-level imports like 'config'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_json_file, find_sources_by_issue_id
import config
from extraction_logic import extract_from_pdf, extract_from_html, merge_and_deduplicate_investors

def load_extraction_definitions():
    """Loads the extraction definitions from the JSON file."""
    # Correctly locate the definitions file relative to this script's location
    definitions_path = os.path.join(os.path.dirname(__file__), 'extraction_definitions.json')
    definitions = load_json_file(definitions_path)
    if not definitions:
        print(f"Error: Could not load extraction definitions from {definitions_path}")
        sys.exit(1)
    return definitions

def main():
    parser = argparse.ArgumentParser(description="Run a specific extraction for a given issue ID based on definitions.")
    parser.add_argument('--issue-id', required=True, help='The issue ID to process.')
    parser.add_argument('--extraction-field', required=True, help='The field to extract (e.g., investors, record_date).')
    args = parser.parse_args()

    issue_id = args.issue_id
    extraction_field = args.extraction_field

    print(f"--- Running extraction for issue '{issue_id}' | field '{extraction_field}' ---")

    # --- Step 1.1: Load and Parse Definitions ---
    definitions = load_extraction_definitions()
    field_definition = definitions.get(extraction_field)

    if not field_definition:
        print(f"Error: No definition found for extraction field '{extraction_field}' in extraction_definitions.json")
        sys.exit(1)

    source_types = field_definition.get("source_types", [])
    semantic_search_query = field_definition.get("semantic_search_query")
    extraction_prompt = field_definition.get("extraction_prompt")
    
    if not all([source_types, extraction_prompt]):
         print(f"Error: Definition for '{extraction_field}' is missing one or more required keys (source_types, extraction_prompt).")
         sys.exit(1)

    print(f"\n[1] Loaded definition for '{extraction_field}'. Target source types: {source_types}")

    # --- Step 1.2: Identify Source Files for the Issue ---
    pdf_sources_data = load_json_file(config.PDF_SOURCES_FILE)
    html_sources_data = load_json_file(config.HTML_SOURCES_FILE)

    pdf_matches, html_matches = find_sources_by_issue_id(issue_id, pdf_sources_data, html_sources_data)
    
    if not pdf_matches and not html_matches:
        print(f"Error: No source documents found for issue_id '{issue_id}'.")
        sys.exit(1)
        
    print(f"[2] Found {len(pdf_matches)} PDF(s) and {len(html_matches)} HTML document(s) for the issue.")

    # --- Phase 2: Source-Specific Extraction ---
    print("\n[3] Starting source-specific extraction process...")
    temp_output_files = []
    
    # Create a temporary directory for individual extraction outputs
    temp_dir = os.path.join(config.OUTPUT_JSON_DIR, 'temp', f"{issue_id}_{extraction_field}")
    os.makedirs(temp_dir, exist_ok=True)

    for doc_type in source_types:
        print(f"\n- Processing source type: '{doc_type}'")
        if doc_type == "PDF":
            for pdf_info in pdf_matches:
                pdf_filename = pdf_info.get("source_url")
                if not pdf_filename:
                    continue
                pdf_path = os.path.join(config.PDF_DIR, pdf_filename)
                
                # Define a unique temporary output path for this document's result
                temp_output_filename = f"{os.path.splitext(pdf_filename)[0]}_{extraction_field}.json"
                temp_output_path = os.path.join(temp_dir, temp_output_filename)

                result_path = extract_from_pdf(
                    pdf_path=pdf_path,
                    search_query=semantic_search_query,
                    extraction_prompt=extraction_prompt,
                    extraction_field=extraction_field,
                    output_path=temp_output_path
                )
                if result_path:
                    temp_output_files.append(result_path)
        else:
            # Assumes other types are source_type from html-sources.json
            for html_info in html_matches:
                if html_info.get("source_type") == doc_type:
                    html_url = html_info.get("source_url")
                    if not html_url:
                        continue
                    
                    # Define a unique temporary output path for this document's result
                    # Create a safe filename from the URL
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

    # --- Phase 3: Merging and Finalization ---
    print(f"\n[4] Merging {len(temp_output_files)} extraction output(s)...")
    
    if temp_output_files:
        final_filename = f"{issue_id}_{extraction_field}_combined.json"
        final_output_path = os.path.join(config.OUTPUT_JSON_DIR, final_filename)
        
        merge_and_deduplicate_investors(
            issue_id=issue_id,
            extraction_field=extraction_field,
            temp_files=temp_output_files,
            final_output_path=final_output_path
        )
        
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"  - Cleaned up temporary directory: {temp_dir}")
    else:
        print("  - No results to merge.")

    print(f"\n--- Extraction for issue '{issue_id}' complete. ---")


if __name__ == "__main__":
    main()
