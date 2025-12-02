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
from utils import load_json_file, find_issue_id, clean_and_parse_json
from typing import List, Dict
from extraction_logic import extract_from_pdf, extract_from_html

# --- Load sources data once for performance ---
SOURCES_DATA = None
HTML_SOURCES_DATA = None

def _load_sources_data():
    """Lazy loader for the sources data to avoid loading it if not needed."""
    global SOURCES_DATA, HTML_SOURCES_DATA
    if SOURCES_DATA is None:
        SOURCES_DATA = load_json_file(config.PDF_SOURCES_FILE)
    if HTML_SOURCES_DATA is None:
        HTML_SOURCES_DATA = load_json_file(config.HTML_SOURCES_FILE)

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

def process_and_save_json(parsed_json: dict, source_path: str, extraction_type: str, source_pages: list):
    """Post-processes and saves the extracted JSON data to a file."""
    # --- Create the final output structure ---
    source_filename_for_lookup = os.path.basename(source_path)
    _load_sources_data() # Ensure source data is loaded
    issue_id = find_issue_id(source_path, SOURCES_DATA, HTML_SOURCES_DATA)

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
    
    # The new unified command uses a different definitions file now.
    # This remains for backward compatibility or direct calls.
    if extraction_type in config.EXTRACTION_QUERIES:
        query = config.EXTRACTION_QUERIES[extraction_type]
        prompt = None # In old system, prompt was derived from extraction_type
    else:
        print(f"\nError: Invalid extraction type '{extraction_type}'.")
        print(f"Available types: {', '.join(config.EXTRACTION_QUERIES.keys())}")
        return

    # Define a final output path for the standalone extraction
    source_filename = os.path.basename(pdf_path)
    json_filename = f"{os.path.splitext(source_filename)[0]}_{extraction_type}.json"
    output_path = os.path.join(config.OUTPUT_JSON_DIR, json_filename)
    
    # Call the refactored logic
    # The prompt is now passed directly. We set it to None here to maintain
    # the old behavior where the vision module derives it from extraction_type.
    extract_from_pdf(pdf_path, query, None, extraction_type, output_path)


def extract_html_command(args):
    """Extract structured data from an HTML file."""
    extraction_type = args.extraction_type
    html_path = args.html_path
    
    # This remains for backward compatibility or direct calls.
    if extraction_type not in config.EXTRACTION_QUERIES:
        print(f"\nError: Invalid extraction type '{extraction_type}'.")
        return

    # Define a final output path for the standalone extraction
    source_filename = os.path.basename(html_path)
    json_filename = f"{os.path.splitext(source_filename)[0]}_{extraction_type}.json"
    output_path = os.path.join(config.OUTPUT_JSON_DIR, json_filename)
    
    # In the old system, the prompt was derived from the type. For the refactored
    # function, we'd ideally pass the full prompt text. This part may need
    # adjustment if this command is intended to be used with the new definitions.
    # For now, we pass None and assume the vision module can handle it.
    extract_from_html(html_path, None, extraction_type, output_path)


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
