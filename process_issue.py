#!/usr/bin/env python3
"""
A script to process all documents associated with a given issue_id.

This script finds all PDF and HTML documents for a specific issue_id,
runs the appropriate extraction on each, and then combines the results.
"""

import json
import sys
import os
import subprocess

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

def main():
    if len(sys.argv) < 3:
        print("Usage: python process_issue.py <issue_id> <extraction_type>")
        print("\nExample:")
        print("  python process_issue.py 96065bbc-05c7-40b2-b7b9-75c11fc7ddb9 underwriters")
        sys.exit(1)
        
    issue_id_to_process = sys.argv[1]
    extraction_type = sys.argv[2]
    
    # Define paths to the source files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_sources_path = os.path.join(base_dir, 'tests', 'sources.json')
    html_sources_path = os.path.join(base_dir, 'tests', 'html-sources.json')
    
    # Load the source files
    pdf_sources = load_json_file(pdf_sources_path)
    html_sources = load_json_file(html_sources_path)
    
    if not pdf_sources and not html_sources:
        print("Could not load any source files. Exiting.")
        sys.exit(1)
        
    pdf_matches, html_matches = find_sources_by_issue_id(issue_id_to_process, pdf_sources, html_sources)
    
    print(f"\nFound {len(pdf_matches)} PDF(s) and {len(html_matches)} HTML source(s) for issue_id: {issue_id_to_process}")
    
    if not pdf_matches and not html_matches:
        print("\nNo documents to process. Exiting.")
        sys.exit(1)

    # --- Step 1: Run extractions on all found documents ---
    output_files = []
    
    # Process PDFs
    for pdf in pdf_matches:
        pdf_filename = pdf.get("source_url")
        pdf_path = os.path.join(base_dir, 'pdfs', pdf_filename)
        print(f"\n--- Processing PDF: {pdf_filename} ---")
        
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found at {pdf_path}. Skipping.")
            continue
            
        subprocess.run(["./run.sh", "extract", extraction_type, pdf_path])
        
        # Store the expected output path
        json_filename = f"{os.path.splitext(pdf_filename)[0]}_{extraction_type}.json"
        output_files.append(os.path.join(base_dir, 'output_json', json_filename))

    # Process HTMLs
    for html in html_matches:
        html_url = html.get("source_url")
        print(f"\n--- Processing HTML: {html_url} ---")
        subprocess.run(["./run.sh", "extract-html", extraction_type, html_url])
        
        # Store the expected output path (derive filename from URL)
        url_basename = os.path.basename(html_url)
        json_filename = f"{os.path.splitext(url_basename)[0]}_{extraction_type}.json"
        output_files.append(os.path.join(base_dir, 'output_json', json_filename))
        
    # --- Step 2: Combine the results ---
    print("\n" + "="*60)
    print("Combining extraction results...")
    print("="*60)

    combined_investors = []
    source_files_processed = []

    for file_path in output_files:
        if os.path.exists(file_path):
            print(f"Reading from {os.path.basename(file_path)}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                investors = data.get("investors", [])
                if investors:
                    combined_investors.extend(investors)
                    source_files_processed.append(os.path.basename(file_path))
        else:
            print(f"Warning: Output file not found at {file_path}. Skipping.")

    # Remove duplicates based on a combination of name and level
    unique_investors = []
    seen_entries = set()
    for investor in combined_investors:
        name = investor.get("name")
        level = investor.get("level")
        
        # Create a unique key for the combination of name and level
        unique_key = (name, level)
        
        if name and unique_key not in seen_entries:
            unique_investors.append(investor)
            seen_entries.add(unique_key)

    # Final combined output
    final_output = {
        "issue_id": issue_id_to_process,
        "investors": unique_investors,
        "source_files": source_files_processed
    }

    # Save the combined file
    combined_filename = f"{issue_id_to_process}_{extraction_type}_combined.json"
    combined_output_path = os.path.join(base_dir, 'output_json', combined_filename)
    
    with open(combined_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccessfully combined {len(unique_investors)} unique investors from {len(source_files_processed)} source file(s).")
    print(f"Combined data saved to: {combined_output_path}")

if __name__ == "__main__":
    main()
