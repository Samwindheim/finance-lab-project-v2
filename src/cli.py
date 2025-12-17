"""
Developer Toolkit for the PDF RAG Pipeline.

This script provides utility commands for development, testing, and maintenance of the
extraction system. It is NOT for production extractions - use `run_extraction.py` for that.

Available Commands:
- `index`: Manually index a single PDF file to build/rebuild its FAISS vector index.
- `query`: Test semantic search queries on a PDF to see which pages are returned.
- `clear`: Delete the FAISS index for a specific PDF.
- `extract-html-text`: Extract and display the text content from an HTML file/URL.

For production data extraction based on `extraction_definitions.json`, use:
  python src/run_extraction.py --issue-id <issue_id> [--extraction-field <field>]
"""

import argparse
import sys
import os
from pdf_indexer import PDFIndexer
from html_processor import extract_text_from_html
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
        description="Developer Toolkit - Utilities for testing and maintaining the PDF RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a PDF (creates a specific index for this PDF)
  ./run.sh index pdfs/document.pdf
  
  # Query the index for a specific PDF to test semantic search
  ./run.sh query pdfs/document.pdf "Find underwriter section" -n 5

  # Extract and display text from an HTML source
  ./run.sh extract-html-text https://example.com/source.html

  # Clear the index for a specific PDF
  ./run.sh clear pdfs/document.pdf

For production extractions, use:
  python src/run_extraction.py --issue-id <issue_id> [--extraction-field <field>]
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
