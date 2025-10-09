#!/usr/bin/env python3
"""
PDF RAG Pipeline CLI
Run this script to index PDFs and query the vector database.
"""

import argparse
import sys
import json
import os
from pdf_indexer import PDFIndexer
from vision import get_json_from_image
import config

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

    results = indexer.query(query, top_k=1)

    if results:
        top_result = results[0]
        print(f"\nExtracting image for top result (Page {top_result['page_number']})...")
        
        # --- Fallback logic for PDF path ---
        # Use the path from metadata if available, otherwise use the path from CLI args
        pdf_path_for_extraction = top_result.get('pdf_path') or args.pdf_path
        
        if not pdf_path_for_extraction:
            print("\nError: Could not determine PDF path for extraction. Please re-index the document or provide a valid path.")
            return

        saved_path = indexer.extract_page_as_image(
            pdf_path=pdf_path_for_extraction,
            page_number=top_result['page_number']
        )
        if saved_path:
            print(f"\nSuccessfully saved image to: {saved_path}")
            json_data = get_json_from_image(saved_path, extraction_type)
            if json_data:
                parsed_json = clean_and_parse_json(json_data)
                if parsed_json:
                    parsed_json['source_pages'] = [top_result['page_number']]
                    pdf_filename = os.path.basename(pdf_path_for_extraction)
                    json_filename = f"{os.path.splitext(pdf_filename)[0]}_{extraction_type}.json"
                    output_dir = config.OUTPUT_JSON_DIR
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, json_filename)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(parsed_json, f, indent=2, ensure_ascii=False)
                    print(f"\nSuccessfully extracted and saved data to: {output_path}\n")
    else:
        print(f"\nCould not find any relevant pages for '{extraction_type}'.")


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
  
  # Query the index for a specific PDF
  ./run.sh query pdfs/document.pdf "Find underwriter section" -n 5

  # Extract underwriter data from a specific PDF
  ./run.sh extract_underwriters pdfs/document.pdf

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
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract structured data from a specific PDF')
    extract_parser.add_argument('extraction_type', help=f"Type of data to extract (e.g., {', '.join(config.EXTRACTION_QUERIES.keys())})")
    extract_parser.add_argument('pdf_path', help='Path to the PDF file')
    extract_parser.set_defaults(func=extract_command)

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
