#!/usr/bin/env python3
"""
Command-Line Interface for the PDF RAG Pipeline.

This script serves as the main user entry point for the entire application. It uses Python's `argparse` to create a robust command-line interface with distinct commands for different operations:
- `index`: Processes a PDF, generates embeddings, and saves them to a FAISS index.
- `query`: Performs a semantic search on an indexed PDF to find relevant text chunks.
- `extract`: A multi-step process that first finds relevant pages using semantic search, converts them to images, and then uses the vision module to extract structured JSON data.
- `clear`: Deletes the FAISS index for a specific PDF.

It coordinates the functionality of the `pdf_indexer` and `vision` modules to execute user requests.
"""

import argparse
import sys
import json
import os
from pdf_indexer import PDFIndexer
from vision import get_json_from_image, get_json_from_text
from html_processor import extract_text_from_html
import config

# --- Load sources data once for performance ---
SOURCES_FILE = os.path.join(config.BASE_DIR, "tests", "sources.json")
SOURCES_DATA = []
if os.path.exists(SOURCES_FILE):
    with open(SOURCES_FILE, 'r') as f:
        SOURCES_DATA = json.load(f)

HTML_SOURCES_FILE = os.path.join(config.BASE_DIR, "tests", "html-sources.json")
HTML_SOURCES_DATA = []
if os.path.exists(HTML_SOURCES_FILE):
    with open(HTML_SOURCES_FILE, 'r') as f:
        HTML_SOURCES_DATA = json.load(f)

def find_issue_id(source_path: str) -> str | None:
    """Finds the issue_id for a given source path from all available source files."""
    # Check if it's a URL or HTML file first
    if source_path.startswith('http') or source_path.lower().endswith(('.html', '.htm')):
        for source in HTML_SOURCES_DATA:
            if source.get("source_url") == source_path:
                # Prioritize issue_id, but fall back to other IDs if needed
                return source.get("issue_id") or source.get("warrant_id") or source.get("convertible_id")
    
    # If not found or it's a PDF, check by filename in the PDF sources
    pdf_filename = os.path.basename(source_path)
    for source in SOURCES_DATA:
        if source.get("source_url") == pdf_filename:
            return source.get("issue_id")
            
    return None

def get_index_path_for_pdf(pdf_path: str, index_dir: str) -> str:
    """Generates a unique index path from a PDF filename."""
    pdf_filename = os.path.basename(pdf_path)
    index_name = os.path.splitext(pdf_filename)[0]
    return os.path.join(index_dir, index_name)

def load_indexer(pdf_path: str, index_dir: str) -> PDFIndexer | None:
    """Loads the PDFIndexer for a given PDF, returns None if index is not found."""
    index_path = get_index_path_for_pdf(pdf_path, index_dir)
    indexer = PDFIndexer(index_path=index_path)
    
    if indexer.index.ntotal == 0:
        print(f"\nIndex for {os.path.basename(pdf_path)} not found or is empty. Please index it first.\n")
        return None
    return indexer

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

def process_and_save_json(parsed_json: dict, source_path: str, extraction_type: str, source_pages: list):
    """Post-processes and saves the extracted JSON data to a file."""
    # --- Create the final output structure ---
    source_filename_for_lookup = os.path.basename(source_path)
    issue_id = find_issue_id(source_path)

    if not issue_id:
        print(f"\nWarning: Could not find issue_id for '{source_filename_for_lookup}' in any source file. The final file will be missing it.")

    # --- Post-process to match manual data format ---
    processed_investors = []
    for investor in parsed_json.get("investors", []):
        # --- Format amount_in_cash ---
        amount = investor.get("amount_in_cash")
        if isinstance(amount, (int, float)):
            investor["amount_in_cash"] = f"{(amount):.3f}"
        
        # --- Format amount_in_percentage ---
        percent = investor.get("amount_in_percentage")
        if isinstance(percent, (int, float)):
            investor["amount_in_percentage"] = str(percent)

        processed_investors.append(investor)

    final_output = {
        "issue_id": issue_id,
        "investors": processed_investors,
        "source_pages": source_pages
    }

    source_filename = os.path.basename(source_path)
    json_filename = f"{os.path.splitext(source_filename)[0]}_{extraction_type}.json"
    output_dir = config.OUTPUT_JSON_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, json_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\nSuccessfully extracted and saved data to: {output_path}\n")

def index_command(args):
    """Index a PDF file."""
    print(f"\n{'='*60}")
    print("Indexing PDF Document")
    print(f"{'='*60}")
    
    index_path = get_index_path_for_pdf(args.pdf_path, args.index_dir)
    indexer = PDFIndexer(index_path=index_path)
    indexer.index_pdf(args.pdf_path)

def query_command(args):
    """Query the vector database."""
    print(f"\n{'='*60}")
    print("Querying Vector Database")
    print(f"{'='*60}")
    
    indexer = load_indexer(args.pdf_path, args.index_dir)
    if not indexer:
        return

    results = indexer.query(args.query, top_k=args.n)
    
    print(f"\nFound {len(results)} relevant pages:\n")
    for result in results:
        print(f"{result['rank']}. Page {result['page_number']} from {result['document_id']}")
        print(f"   Distance: {result['distance']:.4f}")
        print(f"   Preview: {result['text'][:150]}...")

def extract_command(args):
    """Extract structured data from a PDF based on the extraction type."""
    extraction_type = args.extraction_type
    pdf_path = args.pdf_path

    # get the query for the extraction type, defined in config.py
    query = config.EXTRACTION_QUERIES.get(extraction_type)

    if not query:
        print(f"\nError: Invalid extraction type '{extraction_type}'.")
        print(f"Available types: {', '.join(config.EXTRACTION_QUERIES.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"Extracting {extraction_type.title()}")
    print(f"{'='*60}")

    indexer = load_indexer(pdf_path, args.index_dir)
    if not indexer:
        return

    # 1. query the index for top 5 relevant pages
    results = indexer.query(query, top_k=5)

    if results:
        SIMILARITY_DISTANCE_THRESHOLD = 1.05 # 5% threshold

        # --- Page Selection Logic ---
        # Start with the top result and include the second result if its score is close.
        # Then, expand around the top result to find consecutive pages.
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
            pages_to_extract.append({'page_number': current_page}) # Synthetic page
        
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
        pages_to_extract = unique_pages

        # Sort the collected pages by page number to ensure correct order
        pages_to_extract.sort(key=lambda p: p['page_number'])

        page_numbers = [p['page_number'] for p in pages_to_extract]
        print(f"\nFound {len(page_numbers)} relevant pages to analyze: {page_numbers}")

        # --- Fallback logic for PDF path ---
        # Use the path from metadata if available, otherwise use the path from CLI args
        # We check the top result, assuming all consecutive pages are from the same PDF
        pdf_path_for_extraction = top_result.get('pdf_path') or args.pdf_path
        
        if not pdf_path_for_extraction:
            print("\nError: Could not determine PDF path for extraction. Please re-index the document or provide a valid path.")
            return

        # 2. extract the images and text for all identified pages
        saved_image_paths = []
        page_texts = []
        for page in pages_to_extract:
            print(f"Extracting image and text for page {page['page_number']}...")
            
            # Extract image
            saved_path = indexer.extract_page_as_image(
                pdf_path=pdf_path_for_extraction,
                page_number=page['page_number']
            )
            if saved_path:
                saved_image_paths.append(saved_path)

            # Extract text
            text = indexer.get_text_for_page(
                pdf_path=pdf_path_for_extraction,
                page_number=page['page_number']
            )
            if text:
                page_texts.append(text)
        
        if not saved_image_paths:
            print("\nError: Could not extract any images for the identified pages.")
            return

        combined_text = "\n\n--- Page Separator ---\n\n".join(page_texts)
        # 3. get the json data from the images & text using the vision model
        json_data = get_json_from_image(saved_image_paths, combined_text, extraction_type)
        if json_data:
            parsed_json = clean_and_parse_json(json_data)
            if parsed_json:
                process_and_save_json(parsed_json, pdf_path_for_extraction, extraction_type, page_numbers)
    else:
        print(f"\nCould not find any relevant pages for '{extraction_type}'.")


def extract_html_command(args):
    """Extract structured data from an HTML file."""
    extraction_type = args.extraction_type
    html_path = args.html_path

    print(f"\n{'='*60}")
    print(f"Extracting {extraction_type.title()} from HTML")
    print(f"{'='*60}")

    # 1. Extract text from the HTML file
    print(f"Extracting text from {os.path.basename(html_path)}...")
    text = extract_text_from_html(html_path)

    if not text:
        print("\nError: Could not extract any text from the HTML file.")
        return

    # 2. Get the json data from the text using the vision model (text-only)
    json_data = get_json_from_text(text, extraction_type)
    if json_data:
        parsed_json = clean_and_parse_json(json_data)
        if parsed_json:
            process_and_save_json(parsed_json, html_path, extraction_type, [1])


def unified_extract_command(args):
    """
    Extracts structured data from a PDF or HTML source.
    Automatically determines the correct extraction method.
    """
    source = args.source
    extraction_type = args.extraction_type
    
    # --- Determine source type ---
    is_pdf = source.lower().endswith('.pdf')
    is_html = source.lower().startswith('http') or source.lower().endswith(('.html', '.htm'))
    
    if is_pdf:
        # To call the PDF extraction logic, we need to simulate the 'args' object
        # that the original extract_command expects.
        pdf_args = argparse.Namespace(
            extraction_type=extraction_type,
            pdf_path=source,
            index_dir=args.index_dir  # Pass along the index directory
        )
        extract_command(pdf_args)
    elif is_html:
        # Simulate the 'args' object for the HTML extraction logic.
        html_args = argparse.Namespace(
            extraction_type=extraction_type,
            html_path=source
        )
        extract_html_command(html_args)
    else:
        print(f"\nError: Could not determine file type for '{source}'.")
        print("Please provide a PDF file, an HTML file, or a URL.")


def extract_html_text_command(args):
    """Extracts and prints the text content from an HTML file."""
    html_path = args.html_path

    print(f"\n{'='*60}")
    print(f"Extracting text from HTML source: {os.path.basename(html_path)}")
    print(f"{'='*60}\n")

    text = extract_text_from_html(html_path)

    if text:
        print(text)
    else:
        print("\nCould not extract text or the file is empty.")


def clear_command(args):
    """Clear the vector database."""
    print(f"\n{'='*60}")
    print("Clearing Vector Database")
    print(f"{'='*60}")
    
    index_path = get_index_path_for_pdf(args.pdf_path, args.index_dir)
    
    # Check if index files exist before attempting to clear
    if not os.path.exists(f"{index_path}.index"):
        print(f"\nIndex for '{args.pdf_path}' not found.")
        return

    if not args.yes:
        response = input(f"\nAre you sure you want to delete the index for '{args.pdf_path}'? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print(f"\nCancelled.")
            return
    
    indexer = PDFIndexer(index_path=index_path)
    indexer.reset_index()
    print(f"\nIndex for {args.pdf_path} cleared!")


def main():
    parser = argparse.ArgumentParser(
        description="PDF RAG Pipeline - Index and query PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a PDF (creates a specific index for this PDF)
  ./run.sh index pdfs/document.pdf
  
  # Query the index for a specific PDF, n is the number of top results to return
  ./run.sh query pdfs/document.pdf "Find underwriter section" -n 5

  # Extract structured data from any source (PDF or HTML)
  ./run.sh extract underwriters pdfs/document.pdf
  ./run.sh extract underwriters https://example.com/source.html

  # Clear the index for a specific PDF
  ./run.sh clear pdfs/document.pdf
        """
    )
    
    parser.add_argument(
        '--index-dir',
        default='faiss_index',
        help='Directory to store FAISS indexes (default: faiss_index)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index a PDF file')
    index_parser.add_argument('pdf_path', help='Path to the PDF file')
    index_parser.set_defaults(func=index_command)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the vector database for a specific PDF')
    query_parser.add_argument('pdf_path', help='Path to the PDF file to query')
    query_parser.add_argument('query', help='Search query')
    query_parser.add_argument('-n', '--n', type=int, default=3, help='Number of results (default: 3)')
    query_parser.set_defaults(func=query_command)
    
    # Unified Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract structured data from a PDF or HTML source')
    extract_parser.add_argument('extraction_type', help=f"Type of data to extract (e.g., {', '.join(config.EXTRACTION_QUERIES.keys())})")
    extract_parser.add_argument('source', help='Path or URL to the source file (PDF or HTML)')
    extract_parser.set_defaults(func=unified_extract_command)

    # Extract HTML Text command
    extract_html_text_parser = subparsers.add_parser('extract-html-text', help='Extract text from an HTML file and print it')
    extract_html_text_parser.add_argument('html_path', help='Path or URL to the HTML file')
    extract_html_text_parser.set_defaults(func=extract_html_text_command)

    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear the vector database for a specific PDF')
    clear_parser.add_argument('pdf_path', help='Path to the PDF file to clear')
    clear_parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation prompt')
    clear_parser.set_defaults(func=clear_command)
    
    args = parser.parse_args()
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
