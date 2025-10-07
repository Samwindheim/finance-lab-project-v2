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
from vision_experiment import get_json_from_image

# The semantic query to find the underwriter data, have it here for easy modification.
underwriter_query = "Tabell med teckningsåtaganden och garantiåtaganden, med namn, belopp, andel % och totalsumma i SEK"

def index_command(args):
    """Index a PDF file."""
    print(f"\n{'='*60}")
    print("Indexing PDF Document")
    print(f"{'='*60}\n")
    
    indexer = PDFIndexer(index_path=args.index)
    num_pages = indexer.index_pdf(args.pdf_path)
    
    print(f"\n✅ Successfully indexed {num_pages} pages!")
    print(f"📁 Index saved to: {args.index}")


def query_command(args):
    """Query the vector database."""
    print(f"\n{'='*60}")
    print("Querying Vector Database")
    print(f"{'='*60}\n")
    
    indexer = PDFIndexer(index_path=args.index)
    
    if indexer.index.ntotal == 0:
        print("❌ No documents indexed yet. Please index a PDF first.")
        return
    
    results = indexer.query(args.query, top_k=args.n)
    
    print(f"\n📊 Found {len(results)} relevant pages:\n")
    for result in results:
        print(f"{result['rank']}. 📄 Page {result['page_number']} from {result['document_id']}")
        print(f"   📏 Distance: {result['distance']:.4f}")
        print(f"   📝 Preview: {result['text'][:150]}...")
        print()

def extract_underwriters_command(args):
    """Extract underwriters from a PDF."""
    print(f"\n{'='*60}")
    print("Extracting Underwriters")
    print(f"{'='*60}\n")

    indexer = PDFIndexer(index_path=args.index)

    if indexer.index.ntotal == 0:
        print("❌ No documents indexed yet. Please index a PDF first.")
        return

    query = underwriter_query
    results = indexer.query(query, top_k=1)

    if results:
        top_result = results[0]
        print(f"Extracting image for top result (Page {top_result['page_number']})...")
        saved_path = indexer.extract_page_as_image(
            document_id=top_result['document_id'],
            page_number=top_result['page_number']
        )
        if saved_path:
            print(f" \nSuccessfully saved image to: {saved_path}")
            json_data = get_json_from_image(saved_path)
            if json_data:
                # Clean the response to ensure it's valid JSON
                if json_data.startswith("```json"):
                    json_data = json_data[7:-3].strip()
                elif json_data.startswith("```"):
                    json_data = json_data[3:-3].strip()

                try:
                    parsed_json = json.loads(json_data)

                    # Add the source page from the vector search result
                    parsed_json['source_pages'] = [top_result['page_number']]

                    pdf_path = top_result['document_id']
                    pdf_filename = os.path.basename(pdf_path)
                    json_filename = os.path.splitext(pdf_filename)[0] + '.json'

                    output_dir = 'output_json'
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, json_filename)

                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(parsed_json, f, indent=2, ensure_ascii=False)
                    
                    print(f"✓ Successfully extracted and saved data to: {output_path}\n")

                except json.JSONDecodeError:
                    print("❌ Error: Failed to decode JSON from model response.")
                    print("Raw response:")
                    print(json_data)
    else:
        print("❌ Could not find any relevant pages for underwriters.")


def clear_command(args):
    """Clear the vector database."""
    print(f"\n{'='*60}")
    print("Clearing Vector Database")
    print(f"{'='*60}\n")
    
    if not args.yes:
        response = input(f"⚠️  Are you sure you want to delete the index at '{args.index}'? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Cancelled.")
            return
    
    indexer = PDFIndexer(index_path=args.index)
    indexer.reset_index()
    print(f"\n✅ Vector database cleared!")


def main():
    parser = argparse.ArgumentParser(
        description="PDF RAG Pipeline - Index and query PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a PDF
  ./run.sh index pdfs/document.pdf
  
  # Query the database for top 5 results
  ./run.sh query "Find underwriter section" -n 5

  # Extract underwriter data
  ./run.sh extract_underwriters

  # Clear the database
  ./run.sh clear
        """
    )
    
    parser.add_argument(
        '--index',
        default='faiss_index/index',
        help='Path to the FAISS index (default: faiss_index/index)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index a PDF file')
    index_parser.add_argument('pdf_path', help='Path to the PDF file')
    index_parser.set_defaults(func=index_command)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the vector database')
    query_parser.add_argument('query', help='Search query')
    query_parser.add_argument('-n', '--n', type=int, default=3, help='Number of results (default: 3)')
    query_parser.set_defaults(func=query_command)
    
    # Extract Underwriters command
    extract_parser = subparsers.add_parser('extract_underwriters', help='Extract underwriter information from the document')
    extract_parser.set_defaults(func=extract_underwriters_command)

    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear the vector database')
    clear_parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation prompt')
    clear_parser.set_defaults(func=clear_command)
    
    args = parser.parse_args()
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
