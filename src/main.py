"""
Unified Command-Line Interface for the Finance Lab Project.

This script combines the production extraction pipeline with development utilities 
(indexing, querying, etc.) into a single entry point.
"""

import argparse
import sys
import os
import shutil
import json
from logger import setup_logger

logger = setup_logger(__name__)

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils import load_json_file, find_sources_by_issue_id, find_issue_id
from extraction_logic import extract_from_pdf, extract_from_html, merge_and_finalize_outputs
from models import ExtractionDefinitions, ExtractionDefinition
from pdf_indexer import PDFIndexer
from html_processor import extract_text_from_html

# --- Extraction Logic (formerly in run_extraction.py) ---

def load_extraction_definitions():
    """Loads and validates the extraction definitions from the JSON file."""
    definitions_path = os.path.join(os.path.dirname(__file__), 'extraction_definitions.json')
    with open(definitions_path, 'r', encoding='utf-8') as f:
        raw_definitions = json.load(f)
    
    try:
        validated = ExtractionDefinitions.model_validate(raw_definitions)
        return validated.root
    except Exception as e:
        logger.error(f"Extraction definitions in {definitions_path} failed validation: {e}")
        sys.exit(1)

def run_single_extraction(issue_id: str, extraction_field: str, definitions: dict[str, ExtractionDefinition], pdf_matches: list, html_matches: list):
    """Runs the full extraction and merge pipeline for a single field."""
    field_definition = definitions.get(extraction_field)
    if not field_definition:
        logger.warning(f"No definition found for '{extraction_field}'. Skipping.")
        return

    # --- Step 1: Load the Prompt Text from its File ---
    prompt_filename = f"{extraction_field}.txt"
    prompt_path = os.path.join(config.PROMPTS_DIR, prompt_filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            extraction_prompt = f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found for '{extraction_field}'. Please create '{prompt_filename}' in the prompts directory.")
        return

    # --- Step 2: Get Definition Details ---
    source_types = field_definition.source_types
    semantic_search_query = field_definition.semantic_search_query
    page_selection_strategy = field_definition.page_selection_strategy
    
    logger.info(f"Loaded definition for '{extraction_field}'. Target source types: {source_types}")

    # --- Step 3: Source-Specific Extraction ---
    temp_output_files = []
    temp_dir = os.path.join(config.OUTPUT_JSON_DIR, 'temp', f"{issue_id}_{extraction_field}")
    os.makedirs(temp_dir, exist_ok=True)
    
    data_found = False

    for doc_type in source_types:
        if data_found:
            logger.info(f"Data already extracted for '{extraction_field}'. Skipping remaining source types.")
            break
        
        logger.info(f"Looking for '{doc_type}' documents for field '{extraction_field}'...")
            
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
                    break
        
        if data_found:
            break
        
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
                    break

    # --- Step 4: Merging and Finalization ---
    if temp_output_files:
        logger.info(f"Merging {len(temp_output_files)} extraction output(s) for '{extraction_field}'...")
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
        logger.info(f"No results to merge for field '{extraction_field}'.")

def extract_command(args):
    """Handles the 'extract' command for production data extraction."""
    source_link = args.source_link
    issue_id = args.issue_id
    extraction_field = args.extraction_field

    pdf_sources_data = load_json_file(config.PDF_SOURCES_FILE)
    html_sources_data = load_json_file(config.HTML_SOURCES_FILE)

    if source_link:
        detected_id = find_issue_id(source_link, pdf_sources_data, html_sources_data)
        if not detected_id and not issue_id:
            logger.error(f"Could not find an issue_id associated with '{source_link}'. Please provide --issue-id manually.")
            sys.exit(1)
        issue_id = issue_id or detected_id
        target_document = source_link
        logger.info(f"Document mode: '{source_link}' | Issue ID: '{issue_id}'")
    elif issue_id:
        target_document = None
        logger.info(f"Issue mode: '{issue_id}'")
    else:
        logger.error("Please provide either a document link (positional) or an --issue-id.")
        return

    definitions = load_extraction_definitions()

    fields_to_process = []
    if extraction_field:
        if extraction_field in definitions:
            fields_to_process.append(extraction_field)
        else:
            logger.error(f"Extraction field '{extraction_field}' not found in definitions.")
            sys.exit(1)
    else:
        fields_to_process = list(definitions.keys())
        logger.info(f"Found {len(fields_to_process)} fields to extract: {', '.join(fields_to_process)}")

    logger.info("Identifying source documents...")
    pdf_matches, html_matches = find_sources_by_issue_id(issue_id, pdf_sources_data, html_sources_data)

    if target_document:
        target_filename = os.path.basename(target_document)
        pdf_matches = [
            m for m in pdf_matches 
            if target_document == m.get("id") or target_document == m.get("full_url") or target_filename == m.get("source_url") or target_document in m.get("source_url", "")
        ]
        html_matches = [
            m for m in html_matches 
            if target_document == m.get("id") or target_document == m.get("source_url") or target_document == m.get("full_url") or target_document in m.get("source_url", "")
        ]
        logger.info(f"Filtering for document: '{target_document}'")

    if not pdf_matches and not html_matches:
        logger.error(f"No source documents found for issue_id '{issue_id}'" + (f" and document '{target_document}'" if target_document else "") + ". Aborting.")
        sys.exit(1)
    
    logger.info(f"Found {len(pdf_matches)} PDF(s) and {len(html_matches)} HTML document(s) for the issue.")

    for field in fields_to_process:
        logger.info(f"Processing field: '{field}'")
        run_single_extraction(issue_id, field, definitions, pdf_matches, html_matches)

    logger.info(f"All processing for issue '{issue_id}' complete.")

# --- Developer Utilities (formerly in cli.py) ---

def get_index_path_for_pdf(pdf_path: str, index_dir: str) -> str:
    pdf_filename = os.path.basename(pdf_path)
    index_name = os.path.splitext(pdf_filename)[0]
    return os.path.join(index_dir, index_name)

def index_command(args):
    logger.info(f"Indexing PDF Document: {args.pdf_path}")
    index_path = get_index_path_for_pdf(args.pdf_path, args.index_dir)
    indexer = PDFIndexer(index_path=index_path)
    indexer.index_pdf(args.pdf_path)

def query_command(args):
    logger.info(f"Querying Vector Database for: {args.pdf_path}")
    index_path = get_index_path_for_pdf(args.pdf_path, args.index_dir)
    indexer = PDFIndexer(index_path=index_path)
    if indexer.index.ntotal == 0:
        logger.error(f"Index for {os.path.basename(args.pdf_path)} not found. Please index it first.")
        return
    results = indexer.query(args.query, top_k=args.n)
    logger.info(f"Found {len(results)} relevant pages:")
    for result in results:
        logger.info(f"{result['rank']}. Page {result['page_number']} | Distance: {result['distance']:.4f}")
        logger.info(f"   Preview: {result['text'][:150]}...")

def clear_command(args):
    logger.info(f"Clearing Vector Database for: {args.pdf_path}")
    index_path = get_index_path_for_pdf(args.pdf_path, args.index_dir)
    if not os.path.exists(f"{index_path}.index"):
        logger.warning(f"Index for '{args.pdf_path}' not found.")
        return
    if not args.yes:
        response = input(f"Delete index for '{args.pdf_path}'? (y/n): ")
        if response.lower() not in ['y', 'yes']: return
    indexer = PDFIndexer(index_path=index_path)
    indexer.reset_index()
    logger.info("Index cleared!")

def html_text_command(args):
    logger.info(f"Extracting HTML Text: {os.path.basename(args.html_path)}")
    text = extract_text_from_html(args.html_path)
    if text:
        logger.info("\n" + text)
    else:
        logger.error("Could not extract text.")

# --- Main CLI Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Finance Lab Data Extraction & Developer Toolkit")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Extract
    extract_parser = subparsers.add_parser('extract', help='Run production data extraction')
    extract_parser.add_argument('source_link', nargs='?', help='Document filename or URL')
    extract_parser.add_argument('--issue-id', help='Issue ID (auto-detected if link provided)')
    extract_parser.add_argument('--extraction-field', help='Specific field to extract')
    extract_parser.set_defaults(func=extract_command)
    
    # Index
    index_parser = subparsers.add_parser('index', help='Index a PDF file')
    index_parser.add_argument('pdf_path', help='Path to PDF')
    index_parser.add_argument('--index-dir', default='faiss_index', help='Index storage directory')
    index_parser.set_defaults(func=index_command)
    
    # Query
    query_parser = subparsers.add_parser('query', help='Query a PDF index')
    query_parser.add_argument('pdf_path', help='Path to PDF')
    query_parser.add_argument('query', help='Search query')
    query_parser.add_argument('-n', '--n', type=int, default=3, help='Number of results')
    query_parser.add_argument('--index-dir', default='faiss_index', help='Index storage directory')
    query_parser.set_defaults(func=query_command)
    
    # Clear
    clear_parser = subparsers.add_parser('clear', help='Clear a PDF index')
    clear_parser.add_argument('pdf_path', help='Path to PDF')
    clear_parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation')
    clear_parser.add_argument('--index-dir', default='faiss_index', help='Index storage directory')
    clear_parser.set_defaults(func=clear_command)
    
    # HTML Text
    html_parser = subparsers.add_parser('html-text', help='Extract text from HTML')
    html_parser.add_argument('html_path', help='Path or URL')
    html_parser.set_defaults(func=html_text_command)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

