#!/usr/bin/env python3
"""
PDF RAG Pipeline CLI
Run this script to index PDFs and query the vector database.
"""

import argparse
import sys
import json
from pdf_indexer import PDFIndexer
from vision_experiment import get_json_from_image


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
    
    results = indexer.query(args.query, top_k=args.top_k)
    
    print(f"\n📊 Found {len(results)} relevant pages:\n")
    for result in results:
        print(f"{result['rank']}. 📄 Page {result['page_number']} from {result['document_id']}")
        print(f"   📏 Distance: {result['distance']:.4f}")
        print(f"   📝 Preview: {result['text'][:150]}...")
        print()

    # Save image of the top result if requested
    if args.save_image and results:
        top_result = results[0]
        print("\n" + "-"*60)
        print(f"📸 Extracting image for top result (Page {top_result['page_number']})...")
        saved_path = indexer.extract_page_as_image(
            document_id=top_result['document_id'],
            page_number=top_result['page_number']
        )
        if saved_path:
            print(f"🖼️  Successfully saved image to: {saved_path}")
            json_data = get_json_from_image(saved_path)
            if json_data:
                print("\n--- Gemini Response ---")
                print(json_data)
                print("---------------------\n")
        print("-" * 60)


def interactive_command(args):
    """Interactive query mode."""
    print(f"\n{'='*60}")
    print("Interactive Query Mode")
    print(f"{'='*60}\n")
    
    indexer = PDFIndexer(index_path=args.index)
    
    if indexer.index.ntotal == 0:
        print("❌ No documents indexed yet. Please index a PDF first.")
        return
    
    print(f"📚 Loaded index with {indexer.index.ntotal} pages")
    print("💡 Type your queries below (or 'quit' to exit)\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            results = indexer.query(query, top_k=args.top_k)
            
            print(f"\n📊 Top {len(results)} results:\n")
            for result in results:
                print(f"{result['rank']}. Page {result['page_number']} (distance: {result['distance']:.4f})")
                print(f"   {result['text'][:120]}...")
                print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


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


def info_command(args):
    """Show information about the current index."""
    print(f"\n{'='*60}")
    print("Vector Database Info")
    print(f"{'='*60}\n")
    
    indexer = PDFIndexer(index_path=args.index)
    
    print(f"📁 Index path: {args.index}")
    print(f"📊 Total indexed pages: {indexer.index.ntotal}")
    print(f"🔢 Embedding dimension: {indexer.embedding_dimension}")
    print(f"🤖 Embedding model: {indexer.embedding_model}")
    
    if indexer.metadata:
        # Count unique documents
        unique_docs = set(meta['document_id'] for meta in indexer.metadata)
        print(f"📄 Unique documents: {len(unique_docs)}")
        print(f"\nDocuments:")
        for doc_id in unique_docs:
            pages = [m['page_number'] for m in indexer.metadata if m['document_id'] == doc_id]
            print(f"  • {doc_id} ({len(pages)} pages)")
    print()


def list_chunks_command(args):
    """List all indexed chunks and their metadata."""
    print(f"\n{'='*60}")
    print("Listing Indexed Chunks")
    print(f"{'='*60}\n")
    
    indexer = PDFIndexer(index_path=args.index)
    
    if not indexer.metadata:
        print("❌ No chunks found in the index.")
        return
    
    total_chunks = len(indexer.metadata)
    print(f"📄 Found {total_chunks} chunks in the index.\n")
    
    # Determine the number of chunks to show
    limit = args.limit if args.limit is not None and args.limit < total_chunks else total_chunks
    
    for i, chunk_meta in enumerate(indexer.metadata[:limit]):
        print(f"--- Chunk {i+1} ---")
        print(f"  📄 Document ID: {chunk_meta['document_id']}")
        print(f"  🔢 Page Number: {chunk_meta['page_number']}")
        print(f"  📝 Text Preview: {chunk_meta['text'][:200].replace(chr(10), ' ')}...")
        print()

    if args.limit and total_chunks > args.limit:
        print(f"... and {total_chunks - args.limit} more chunks.")


def main():
    parser = argparse.ArgumentParser(
        description="PDF RAG Pipeline - Index and query PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a PDF
  python run.py index pdfs/document.pdf
  
  # Query the database
  python run.py query "Find underwriter section"
  
  # Interactive mode
  python run.py interactive
  
  # Show index info
  python run.py info
  
  # List indexed chunks
  ./run.sh list-chunks -n 5

  # Clear the database
  python run.py clear
        """
    )
    
    parser.add_argument(
        '--index',
        default='faiss_index',
        help='Path to the FAISS index (default: faiss_index)'
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
    query_parser.add_argument('-k', '--top-k', type=int, default=3, help='Number of results (default: 3)')
    query_parser.add_argument('--save-image', action='store_true', help='Save an image of the top matching page')
    query_parser.set_defaults(func=query_command)
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive query mode')
    interactive_parser.add_argument('-k', '--top-k', type=int, default=3, help='Number of results (default: 3)')
    interactive_parser.set_defaults(func=interactive_command)
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear the vector database')
    clear_parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation prompt')
    clear_parser.set_defaults(func=clear_command)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show index information')
    info_parser.set_defaults(func=info_command)

    # List Chunks command
    list_parser = subparsers.add_parser('list-chunks', help='List all indexed chunks and their metadata')
    list_parser.add_argument('-n', '--limit', type=int, help='Limit the number of chunks to display')
    list_parser.set_defaults(func=list_chunks_command)
    
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
