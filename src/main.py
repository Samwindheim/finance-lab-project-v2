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
from utils import load_json_file, find_sources_by_issue_id, find_issue_id, download_pdf
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
                
                # Check if file exists locally, if not download it
                if not os.path.exists(pdf_path):
                    full_url = pdf_info.get("full_url")
                    if full_url:
                        logger.info(f"PDF {pdf_filename} not found locally. Attempting to download...")
                        success = download_pdf(full_url, pdf_path)
                        if not success:
                            logger.error(f"Could not download {pdf_filename}. Skipping.")
                            continue
                    else:
                        logger.warning(f"PDF {pdf_filename} not found and no URL available. Skipping.")
                        continue

                doc_id = pdf_info.get("id", "unknown_pdf")
                temp_output_filename = f"{doc_id}_{extraction_field}.json"
                temp_output_path = os.path.join(temp_dir, temp_output_filename)

                result_path = extract_from_pdf(
                    pdf_path=pdf_path,
                    search_query=semantic_search_query,
                    extraction_prompt=extraction_prompt,
                    extraction_field=extraction_field,
                    output_path=temp_output_path,
                    page_selection_strategy=page_selection_strategy,
                    issue_id=issue_id
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
                
                doc_id = html_info.get("id", "unknown_html")
                temp_output_filename = f"{doc_id}_{extraction_field}.json"
                temp_output_path = os.path.join(temp_dir, temp_output_filename)

                result_path = extract_from_html(
                    html_path=html_url,
                    extraction_prompt=extraction_prompt,
                    extraction_field=extraction_field,
                    output_path=temp_output_path,
                    issue_id=issue_id
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

    # --- Step 1: Identify source documents ---
    logger.info("Identifying source documents...")
    pdf_matches, html_matches = find_sources_by_issue_id(issue_id, pdf_sources_data, html_sources_data)

    # Detect the issue type from the first match
    issue_type = None
    all_matches = pdf_matches + html_matches
    if all_matches:
        # Check if any match has an issue_type
        for m in all_matches:
            if m.get("issue_type"):
                issue_type = m.get("issue_type")
                break
    
    if not issue_type:
        logger.warning(f"Could not detect issue_type for issue_id '{issue_id}'. Assuming 'Rights issue'.")
        issue_type = "Rights issue"
    else:
        logger.info(f"Detected issue_type: '{issue_type}'")

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

    # --- Step 2: Determine fields to process (Smart Filtering) ---
    definitions = load_extraction_definitions()
    fields_to_process = []
    
    # Identify the source types of our target document(s)
    target_source_types = set()
    for m in pdf_matches:
        if m.get("source_type"):
            target_source_types.add(m.get("source_type"))
    for m in html_matches:
        if m.get("source_type"):
            target_source_types.add(m.get("source_type"))

    if extraction_field:
        if extraction_field in definitions:
            fields_to_process.append(extraction_field)
        else:
            logger.error(f"Extraction field '{extraction_field}' not found in definitions.")
            sys.exit(1)
    else:
        # Smart Filter: Only include fields that match the target document's source type AND issue type
        for field_name, field_def in definitions.items():
            # Filter by issue type first
            if field_def.issue_types and issue_type not in field_def.issue_types:
                logger.debug(f"Skipping field '{field_name}' - not relevant for issue_type: {issue_type}")
                continue

            if target_document:
                if any(st in field_def.source_types for st in target_source_types):
                    fields_to_process.append(field_name)
                else:
                    logger.debug(f"Skipping field '{field_name}' - not relevant for source types: {target_source_types}")
            else:
                fields_to_process.append(field_name)
        
        if target_document:
            logger.info(f"Smart Filter: Identified {len(fields_to_process)} relevant fields for source types {target_source_types} and issue_type '{issue_type}': {', '.join(fields_to_process)}")
        else:
            logger.info(f"Smart Filter: Identified {len(fields_to_process)} relevant fields for issue_type '{issue_type}': {', '.join(fields_to_process)}")

    if not fields_to_process:
        logger.warning(f"No relevant fields found for processing with source types: {target_source_types}")
        return

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
    pdf_path = args.pdf_path
    
    # Check if file exists, if not try to find it in manifests
    if not os.path.exists(pdf_path):
        pdf_filename = os.path.basename(pdf_path)
        pdf_sources_data = load_json_file(config.PDF_SOURCES_FILE)
        
        # Look for the filename in the manifest
        match = next((m for m in pdf_sources_data if m.get("source_url") == pdf_filename), None)
        if match and match.get("full_url"):
            logger.info(f"PDF {pdf_filename} not found locally. Attempting to download from manifest...")
            if not download_pdf(match["full_url"], pdf_path):
                logger.error(f"Failed to download {pdf_filename}.")
                return
        else:
            logger.error(f"PDF file not found: {pdf_path} (and no download URL found in manifests)")
            return

    logger.info(f"Indexing PDF Document: {pdf_path}")
    index_path = get_index_path_for_pdf(pdf_path, args.index_dir)
    indexer = PDFIndexer(index_path=index_path)
    indexer.index_pdf(pdf_path)

def query_command(args):
    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        # Try to find it in manifests if it's just a filename
        pdf_filename = os.path.basename(pdf_path)
        pdf_sources_data = load_json_file(config.PDF_SOURCES_FILE)
        match = next((m for m in pdf_sources_data if m.get("source_url") == pdf_filename), None)
        if match and match.get("full_url"):
            logger.info(f"PDF {pdf_filename} not found locally. Attempting to download from manifest...")
            if not download_pdf(match["full_url"], pdf_path):
                logger.error(f"Failed to download {pdf_filename}.")
                return
        else:
            logger.error(f"PDF file not found: {pdf_path}")
            return

    logger.info(f"Querying Vector Database for: {pdf_path}")
    index_path = get_index_path_for_pdf(pdf_path, args.index_dir)
    indexer = PDFIndexer(index_path=index_path)
    if indexer.index.ntotal == 0:
        logger.error(f"Index for {os.path.basename(pdf_path)} not found. Please index it first.")
        return
    results = indexer.query(args.query, top_k=args.n)
    logger.info(f"Found {len(results)} relevant pages:")
    for result in results:
        logger.info(f"{result['rank']}. Page {result['page_number']} | Distance: {result['distance']:.4f}")
        logger.info(f"   Preview: {result['text'][:150]}...")

def clear_command(args):
    pdf_path = args.pdf_path
    # No download for clear, just check if it exists or if the index exists
    index_path = get_index_path_for_pdf(pdf_path, args.index_dir)
    if not os.path.exists(f"{index_path}.index"):
        logger.warning(f"Index for '{pdf_path}' not found.")
        return
    
    if not args.yes:
        response = input(f"Delete index for '{pdf_path}'? (y/n): ")
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

def test_command(args):
    """Handles the 'test' command for evaluating extraction accuracy."""
    from tests.run_db_evaluation import main as run_eval
    # Adjust sys.argv so the imported main() can parse them
    sys.argv = ['tests/run_db_evaluation.py', '--issue-id', args.issue_id]
    if args.output_dir:
        sys.argv.extend(['--output-dir', args.output_dir])
    run_eval()

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

    # Test
    test_parser = subparsers.add_parser('test', help='Evaluate AI extraction against SQL Database')
    test_parser.add_argument('--issue-id', required=True, help='The issue_id to evaluate')
    test_parser.add_argument('--output-dir', default='output_json', help='Directory containing AI JSON results')
    test_parser.set_defaults(func=test_command)

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

