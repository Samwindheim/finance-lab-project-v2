"""
[DEPRECATED] Alternative Multi-Source Extraction Pipeline.

This script represents an earlier approach to processing an entire issue from multiple
source documents. It works by creating a single, unified FAISS index for all documents
related to an issue and then performing one large semantic search across all of them.

This approach has been superseded by the pipeline in `run_extraction.py`, which processes
each document individually and then merges the results. The new method was found to be
more reliable and is the standard process going forward.

This file is kept for archival purposes only.
"""

import json
import sys
import os

from source_indexer import SourceIndexer
from vision import get_json_from_text, get_json_from_image
from utils import load_json_file, find_sources_by_issue_id, clean_and_parse_json
from html_processor import extract_text_from_html
from cli import select_consecutive_pages
import config

def save_combined_json(issue_id: str, extraction_type: str, parsed_json: dict, contributing_sources: list):
    """Saves the final combined and processed JSON to a file."""
    
    # The parsed_json from the model should be {"investors": [...]}
    # We wrap it in the final structure with the issue_id and contributing sources.
    final_output = {
        "issue_id": issue_id,
        "investors": parsed_json.get("investors", []),
        "contributing_sources": contributing_sources
    }

    # Save the combined file
    output_dir = config.OUTPUT_JSON_DIR
    os.makedirs(output_dir, exist_ok=True)
    combined_filename = f"{issue_id}_{extraction_type}_combined.json"
    combined_output_path = os.path.join(output_dir, combined_filename)
    
    with open(combined_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccessfully processed and combined data for issue {issue_id}.")
    print(f"Combined data saved to: {combined_output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python src/process_issue.py <issue_id> <extraction_type>")
        print("\nExample:")
        print("  python src/process_issue.py 96065bbc-05c7-40b2-b7b9-75c11fc7ddb9 underwriters")
        sys.exit(1)
        
    issue_id = sys.argv[1]
    extraction_type = sys.argv[2]
    
    # --- Step 1: Initialize the indexer for the issue ---
    indexer = SourceIndexer(issue_id=issue_id)
    
    # --- Step 2: Check if the index exists. If not, build it. ---
    if indexer.index.ntotal == 0:
        print(f"No existing index found for issue {issue_id}. Building a new one...")
        
        # Load all source files to find the documents for this issue
        pdf_sources = load_json_file(config.PDF_SOURCES_FILE)
        html_sources = load_json_file(config.HTML_SOURCES_FILE)
        
        pdf_matches, html_matches = find_sources_by_issue_id(issue_id, pdf_sources, html_sources)
        
        if not pdf_matches and not html_matches:
            print(f"No source documents found for issue {issue_id}. Exiting.")
            sys.exit(1)
            
        indexer.index_issue(pdf_matches, html_matches)

    # --- Step 3: Query the unified index ---
    query_text = config.EXTRACTION_QUERIES.get(extraction_type)
    if not query_text:
        print(f"Error: No query found for extraction type '{extraction_type}' in config.py")
        sys.exit(1)
        
    print(f"\nQuerying the unified index for '{extraction_type}'...")
    results = indexer.query(query_text, top_k=15)

    # --- TEMPORARY TEST CODE, uncomment to see query results---
    # print("\n--- Top 15 Search Results ---")
    # for i, res in enumerate(results):
    #     print(f"\n{i+1}. Distance: {res['distance']:.4f}")
    #     print(f"   Source: {res['source_document']} (Type: {res['source_type']}, Page: {res.get('page_number', 'N/A')})")
    #     print(f"   Text: {res['text'][:200]}...")
    # print("\n" + "="*60)
    # sys.exit(0) # Stop the script here
    # --- END TEMPORARY TEST CODE ---
    
    if not results:
        print("No relevant text chunks found in the unified index.")
        sys.exit(1)
        
    # --- Step 4: Prepare context and call the vision model ---
    final_context_texts = []
    image_paths_to_extract = set() # Use a set to avoid duplicate images
    
    # Separate results by type
    pdf_results = [r for r in results if r.get("source_type") == "PDF"]
    html_results = [r for r in results if r.get("source_type") == "HTML"]
    
    best_pdf_res = pdf_results[0] if pdf_results else None
    best_html_res = html_results[0] if html_results else None
    
    use_html_only = False
    
    # Determine if we should use HTML only
    if best_html_res:
        # If no PDF results found, or HTML is strictly better (lower distance)
        if not best_pdf_res or best_html_res['distance'] < best_pdf_res['distance']:
            use_html_only = True
            print(f"  - HTML result ({best_html_res['distance']:.4f}) is better than best PDF ({best_pdf_res['distance']:.4f if best_pdf_res else 'None'}). Using HTML only.")

    # --- Build Context ---
    if use_html_only:
        # Case 1: Use ONLY the HTML text
        source_doc = best_html_res['source_document']
        print(f"  - Found relevant HTML, adding full text from: {source_doc}")
        # We re-extract to ensure we have the full text (in case embedding used truncated version)
        full_html_text = extract_text_from_html(source_doc, preserve_tables=True)
        if full_html_text:
             final_context_texts.append(f"--- START HTML DOCUMENT: {source_doc} ---\n{full_html_text}\n--- END HTML DOCUMENT ---")
             
    else:
        # Case 2: Standard PDF Logic + Top HTML (if exists)
        
        # Add Top HTML if available
        if best_html_res:
            source_doc = best_html_res['source_document']
            print(f"  - Including top HTML context from: {source_doc}")
            full_html_text = extract_text_from_html(source_doc, preserve_tables=True)
            if full_html_text:
                final_context_texts.append(f"--- START HTML DOCUMENT: {source_doc} ---\n{full_html_text}\n--- END HTML DOCUMENT ---")
        
        # Process PDF Results
        if pdf_results:
            # We need to group results by document before selecting pages
            from collections import defaultdict
            pdf_results_by_doc = defaultdict(list)
            for res in pdf_results:
                pdf_results_by_doc[res.get("source_document")].append(res)

            for pdf_filename, doc_results in pdf_results_by_doc.items():
                # To align with cli.py, we only consider the top 5 results for this document.
                top_5_doc_results = doc_results[:5]
                selected_pages = select_consecutive_pages(top_5_doc_results)
                print(f"  - Selected consecutive pages for {pdf_filename}: {selected_pages}")
                
                for page_num in selected_pages:
                    image_paths_to_extract.add((pdf_filename, page_num))
                
                # Also add the text from these selected pages to the context
                for res in doc_results:
                    if res.get("page_number") in selected_pages:
                        final_context_texts.append(f"--- PDF TEXT CHUNK (Page {res.get('page_number')}) ---\n{res.get('text')}")

    if not final_context_texts:
        print("Could not build a valid context for the language model. Exiting.")
        sys.exit(1)
        
    combined_text = "\n\n--- Context Separator ---\n\n".join(final_context_texts)
    
    # --- Step 6: Extract images for the relevant PDF pages ---
    extracted_image_paths = []
    if image_paths_to_extract:
        print(f"\nExtracting {len(image_paths_to_extract)} images from relevant PDF pages...")
        for pdf_filename, page_num in sorted(list(image_paths_to_extract)):
            img_path = indexer.extract_page_as_image(pdf_filename, page_num)
            if img_path:
                extracted_image_paths.append(img_path)
    
    # --- TEMPORARY DEBUG CODE, saves context to file ---
    with open('context.txt', 'w') as f:
        f.write(combined_text)
        for img_path in sorted(extracted_image_paths):
            f.write("\n")
            f.write(f"- {img_path}\n")

    sys.exit(0) # Stop before calling LLM
    

    # --- END TEMPORARY DEBUG CODE ---
    
    # --- Step 7: Call the appropriate vision model function ---
    print("\nSending combined context to the language model for extraction...")
    if extracted_image_paths:
        # If we have images, use the multi-modal function
        print(f"  - Including {len(extracted_image_paths)} images in the request.")
        json_data = get_json_from_image(extracted_image_paths, combined_text, extraction_type)
    else:
        # Otherwise, use the text-only function
        json_data = get_json_from_text(combined_text, extraction_type)
    
    # --- Step 8: Process and save the final output ---
    if json_data:
        parsed_json = clean_and_parse_json(json_data)
        if parsed_json:
            # Get a unique list of documents that contributed to the context
            contributing_docs = sorted(list(set([res['source_document'] for res in results])))
            save_combined_json(issue_id, extraction_type, parsed_json, contributing_docs)

if __name__ == "__main__":
    main()
