"""
Unified Command-Line Interface.

Combines production extraction (historical/new modes) with developer utilities
(indexing, querying, testing) into a single entry point.
"""

import argparse
import sys
import os
import shutil
import json
import pandas as pd
from sqlalchemy import text
from logger import setup_logger

logger = setup_logger(__name__)

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils import find_sources_by_issue_id, find_issue_id, download_pdf, get_db
from extraction_logic import extract_from_pdf, extract_from_html, merge_and_finalize_outputs, classify_html_document
from models import ExtractionDefinitions
from pdf_indexer import PDFIndexer
from html_processor import extract_text_from_html

def _resolve_fields(extraction_field: str, definitions: dict) -> list:
    """Returns the list of fields to process, validating a specific field if provided."""
    if extraction_field:
        if extraction_field not in definitions:
            logger.error(f"Extraction field '{extraction_field}' not found in definitions.")
            sys.exit(1)
        return [extraction_field]
    return list(definitions.keys())


def load_extraction_definitions():
    """Loads and validates the extraction definitions from the JSON file."""
    definitions_path = os.path.join(os.path.dirname(__file__), 'extraction_definitions.json')
    with open(definitions_path, 'r', encoding='utf-8') as f:
        raw_definitions = json.load(f)
    
    try:
        validated = ExtractionDefinitions.model_validate(raw_definitions)
        return validated
    except Exception as e:
        logger.error(f"Extraction definitions in {definitions_path} failed validation: {e}")
        sys.exit(1)

def _resolve_pdf_path(pdf_info: dict) -> str | None:
    """Resolves a PDF source info dict to a local path, downloading if needed."""
    source_url = pdf_info.get("source_url")
    if not source_url:
        return None
    if source_url.startswith("http"):
        pdf_path = os.path.join(config.PDF_DIR, os.path.basename(source_url))
    else:
        pdf_path = os.path.join(config.PDF_DIR, source_url)
    if not os.path.exists(pdf_path):
        if source_url.startswith("http"):
            logger.info(f"PDF not found locally. Attempting to download: {source_url}")
            if not download_pdf(source_url, pdf_path):
                logger.error(f"Could not download {source_url}. Skipping.")
                return None
        else:
            logger.warning(f"PDF {source_url} not found and no URL available. Skipping.")
            return None
    return pdf_path


def run_single_extraction(issue_id: str, extraction_field: str, definitions_obj: ExtractionDefinitions, pdf_matches: list, html_matches: list, issue_type: str = None, force_unlinked: bool = False):
    """Runs the full extraction and merge pipeline for a single field."""
    field_definition = definitions_obj.field_definitions.get(extraction_field)
    if not field_definition:
        logger.warning(f"No definition found for '{extraction_field}'. Skipping.")
        return

    # Get issue-specific guidance if available
    issue_guidance = ""
    if issue_type and definitions_obj.issue_type_guidance:
        issue_guidance = definitions_obj.issue_type_guidance.get(issue_type, "")

    # --- Step 1: Load the Prompt Text from its File ---
    prompt_filename = f"{extraction_field}.txt"
    prompt_path = os.path.join(config.PROMPTS_DIR, prompt_filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            extraction_prompt = f.read()
            if issue_guidance:
                extraction_prompt = f"{extraction_prompt}\n\nIMPORTANT CONTEXT FOR THIS EXTRACTION:\n{issue_guidance}"
    except FileNotFoundError:
        logger.error(f"Prompt file not found for '{extraction_field}'. Please create '{prompt_filename}' in the prompts directory.")
        return

    # --- Step 2: Get Definition Details ---
    source_types = field_definition.source_types
    semantic_search_query = field_definition.semantic_search_query
    page_selection_strategy = field_definition.page_selection_strategy

    # --- Step 3: Source-Specific Extraction ---
    temp_output_files = []
    temp_dir = os.path.join(config.OUTPUT_JSON_DIR, 'temp', f"{issue_id}_{extraction_field}")
    os.makedirs(temp_dir, exist_ok=True)

    # Filter by source_type unless in unlinked (zero-context) mode
    if force_unlinked:
        # In unlinked mode, we still want to respect the source_type filter
        # if it was explicitly detected/provided.
        active_pdfs = [m for m in pdf_matches if not m.get("source_type") or m.get("source_type") in source_types]
        active_htmls = [m for m in html_matches if not m.get("source_type") or m.get("source_type") in source_types]
    else:
        active_pdfs = [m for m in pdf_matches if m.get("source_type") in source_types]
        active_htmls = [m for m in html_matches if m.get("source_type") in source_types]

    if active_pdfs or active_htmls:
        print("")
        logger.info(f"Extracting field: '{extraction_field}'")

    for pdf_info in active_pdfs:
        pdf_path = _resolve_pdf_path(pdf_info)
        if not pdf_path:
            continue
        doc_id = pdf_info.get("id", "unknown_pdf")
        temp_output_path = os.path.join(temp_dir, f"{doc_id}_{extraction_field}.json")
        result_path = extract_from_pdf(
            pdf_path=pdf_path,
            search_query=semantic_search_query,
            extraction_prompt=extraction_prompt,
            extraction_field=extraction_field,
            output_path=temp_output_path,
            page_selection_strategy=page_selection_strategy,
            issue_id=issue_id,
            source_url=pdf_info.get("source_url")
        )
        if result_path:
            temp_output_files.append(result_path)

    for html_info in active_htmls:
        html_url = html_info.get("source_url")
        if not html_url:
            continue
        doc_id = html_info.get("id", "unknown_html")
        temp_output_path = os.path.join(temp_dir, f"{doc_id}_{extraction_field}.json")
        result_path = extract_from_html(
            html_path=html_url,
            extraction_prompt=extraction_prompt,
            extraction_field=extraction_field,
            output_path=temp_output_path,
            issue_id=issue_id
        )
        if result_path:
            temp_output_files.append(result_path)

    # --- Step 4: Merging and Finalization ---
    if temp_output_files:
        # Use 'unlinked' for the filename if issue_id is None
        final_filename = f"{issue_id or 'unlinked'}_extraction.json"
        final_output_path = os.path.join(config.OUTPUT_JSON_DIR, final_filename)
        
        merge_and_finalize_outputs(
            issue_id=issue_id,
            extraction_field=extraction_field,
            temp_files=temp_output_files,
            final_output_path=final_output_path
        )
        
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def extract_historical_command(args):
    """Handles the 'extract-historical' command for production data extraction."""
    source_link = args.source_link
    issue_id = args.issue_id
    extraction_field = args.extraction_field

    print("  --------------------------------")
    if source_link:
        detected_id = find_issue_id(source_link)
        if not detected_id and not issue_id:
            logger.error(f"Could not find an issue_id associated with '{source_link}' in the database. Please provide --issue-id manually or use 'extract-new'.")
            sys.exit(1)
        issue_id = issue_id or detected_id
        target_document = source_link
        logger.info(f"Historical Mode: '{source_link}' | Issue ID: '{issue_id}'")
    elif issue_id:
        target_document = None
        logger.info(f"Historical Mode: '{issue_id}'")
    else:
        logger.error("Please provide either a document link (positional) or an --issue-id.")
        return

    # --- Step 1: Identify source documents ---
    pdf_matches, html_matches = find_sources_by_issue_id(issue_id)

    if target_document:
        target_filename = os.path.basename(target_document)
        pdf_matches = [
            m for m in pdf_matches
            if target_document == m.get("id") or target_filename == m.get("source_url") or target_document in m.get("source_url", "")
        ]
        html_matches = [
            m for m in html_matches
            if target_document == m.get("id") or target_document == m.get("source_url") or target_document in m.get("source_url", "")
        ]
        logger.info(f"Filtering for document: '{target_document}'")

    if not pdf_matches and not html_matches:
        logger.error(f"No source documents found for issue_id '{issue_id}'" + (f" and document '{target_document}'" if target_document else "") + ". Aborting.")
        sys.exit(1)
    
    # Detect the issue type from the issues table
    issue_data = get_db().get_issue_data(issue_id)
    issue_type = issue_data.get("type") if issue_data else None
    if not issue_type:
        logger.warning(f"Could not detect issue type for issue_id '{issue_id}'. Assuming 'Rights issue'.")
        issue_type = "Rights issue"
    else:
        logger.info(f"Detected issue type: '{issue_type}'")

    logger.info(f"Found {len(pdf_matches)} PDF(s) and {len(html_matches)} HTML document(s) for the issue.")

    # --- Step 2: Determine fields to process (Smart Filtering) ---
    definitions_obj = load_extraction_definitions()
    definitions = definitions_obj.field_definitions
    target_source_types = {m["source_type"] for m in pdf_matches + html_matches if m.get("source_type")}

    if extraction_field:
        fields_to_process = _resolve_fields(extraction_field, definitions)
    else:
        fields_to_process = []
        for field_name, field_def in definitions.items():
            if field_def.issue_types and issue_type not in field_def.issue_types:
                logger.debug(f"Skipping field '{field_name}' - not relevant for issue_type: {issue_type}")
                continue
            if target_document and not any(st in field_def.source_types for st in target_source_types):
                logger.debug(f"Skipping field '{field_name}' - not relevant for source types: {target_source_types}")
                continue
            fields_to_process.append(field_name)

        logger.info(f"Fields to extract: {', '.join(fields_to_process)}")

    if not fields_to_process:
        logger.warning(f"No relevant fields found for processing.")
        return

    for field in fields_to_process:
        run_single_extraction(issue_id, field, definitions_obj, pdf_matches, html_matches, issue_type=issue_type, force_unlinked=False)

    print("")
    logger.info(f"All processing for issue '{issue_id}' complete.")
    print("  --------------------------------")
def extract_new_command(args):
    """Handles the 'extract-new' command for zero-context extraction on new documents."""
    source_link = args.source_link
    print("  --------------------------------")
    logger.info(f"New Document Mode: '{source_link}'")

    if source_link.lower().endswith(".pdf"):
        pdf_matches = [{"source_url": source_link, "id": "new_pdf", "source_type": "Prospectus"}]
        html_matches = []
        issue_type = None
    else:
        logger.info(f"Classifying document...")
        classification = classify_html_document(source_link)
        source_type = classification.source_type or "Publication"
        issue_type = classification.issue_type
        if classification.flags:
            active_flags = [k for k, v in classification.flags.model_dump().items() if v]
            if active_flags:
                logger.warning(f"Flags raised: {', '.join(active_flags)}")
                logger.warning("Aborting extraction")
                print("  --------------------------------")
                return
        logger.info(f"Classification: {source_type} | Issue: {issue_type}")
        pdf_matches = []
        html_matches = [{"source_url": source_link, "id": "new_html", "source_type": source_type}]

    definitions_obj = load_extraction_definitions()
    definitions = definitions_obj.field_definitions
    target_source_types = {m["source_type"] for m in pdf_matches + html_matches if m.get("source_type")}

    if args.extraction_field:
        fields_to_process = _resolve_fields(args.extraction_field, definitions)
    else:
        fields_to_process = []
        for field_name, field_def in definitions.items():
            if field_def.issue_types and issue_type not in field_def.issue_types:
                continue
            if not any(st in field_def.source_types for st in target_source_types):
                continue
            fields_to_process.append(field_name)

        logger.info(f"Identified {len(fields_to_process)} fields to extract: {', '.join(fields_to_process)}")

    for field in fields_to_process:
        run_single_extraction(None, field, definitions_obj, pdf_matches, html_matches, issue_type=issue_type, force_unlinked=False)

    logger.info(f"All processing for new document complete.")
    print("  --------------------------------")
def _ensure_pdf_available(pdf_path: str) -> bool:
    """Ensures a PDF exists locally, downloading from the database record if needed."""
    if os.path.exists(pdf_path):
        return True
    pdf_filename = os.path.basename(pdf_path)
    db = get_db()
    df = pd.read_sql(text("SELECT * FROM sources WHERE source_url LIKE :fn"), db.engine, params={"fn": f"%{pdf_filename}%"})
    if df.empty:
        logger.error(f"PDF file not found: {pdf_path} (and not found in database)")
        return False
    match = df.iloc[0]
    url = match.get("source_url", "")
    if url.startswith("http"):
        logger.info(f"PDF {pdf_filename} not found locally. Attempting to download from database record...")
        return download_pdf(url, pdf_path)
    logger.error(f"PDF file not found: {pdf_path} (and no download URL found in database)")
    return False


def index_command(args):
    pdf_path = args.pdf_path
    if not _ensure_pdf_available(pdf_path):
        return
    logger.info(f"Indexing PDF Document: {pdf_path}")
    indexer = PDFIndexer(index_path=get_index_path_for_pdf(pdf_path, args.index_dir))
    indexer.index_pdf(pdf_path)

def query_command(args):
    pdf_path = args.pdf_path
    if not _ensure_pdf_available(pdf_path):
        return
    logger.info(f"Querying Vector Database for: {pdf_path}")
    indexer = PDFIndexer(index_path=get_index_path_for_pdf(pdf_path, args.index_dir))
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

def batch_test_command(args):
    """Handles the 'batch_test' command for batch evaluation."""
    from tests.batch_test import main as run_batch_test
    # Adjust sys.argv so the imported main() can parse them
    sys.argv = ['tests/batch_test.py']
    if args.n:
        sys.argv.extend(['-n', str(args.n)])
    if args.output_dir:
        sys.argv.extend(['--output-dir', args.output_dir])
    if args.output_file:
        sys.argv.extend(['--output-file', args.output_file])
    if args.exclude_fields:
        sys.argv.extend(['--exclude-fields'] + args.exclude_fields)
    run_batch_test()

def get_index_path_for_pdf(pdf_path: str, index_dir: str) -> str:
    pdf_filename = os.path.basename(pdf_path)
    index_name = os.path.splitext(pdf_filename)[0]
    return os.path.join(index_dir, index_name)

# --- Main CLI Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Finance Lab Data Extraction & Developer Toolkit")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Extract Historical
    hist_parser = subparsers.add_parser('extract-historical', help='Run context-aware extraction on existing documents')
    hist_parser.add_argument('source_link', nargs='?', help='Document filename or URL')
    hist_parser.add_argument('--issue-id', help='Issue ID (auto-detected if link provided)')
    hist_parser.add_argument('--extraction-field', help='Specific field to extract')
    hist_parser.set_defaults(func=extract_historical_command)

    # Extract New
    new_parser = subparsers.add_parser('extract-new', help='Run zero-context extraction on new documents')
    new_parser.add_argument('source_link', help='Document filename or URL')
    new_parser.add_argument('--extraction-field', help='Specific field to extract')
    new_parser.set_defaults(func=extract_new_command)
    
    # Extract (Legacy/Alias for Historical)
    extract_parser = subparsers.add_parser('extract', help='Alias for extract-historical')
    extract_parser.add_argument('source_link', nargs='?', help='Document filename or URL')
    extract_parser.add_argument('--issue-id', help='Issue ID (auto-detected if link provided)')
    extract_parser.add_argument('--extraction-field', help='Specific field to extract')
    extract_parser.set_defaults(func=extract_historical_command)
    
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
    
    # Batch Test
    batch_test_parser = subparsers.add_parser('batch_test', help='Batch evaluate multiple extractions')
    batch_test_parser.add_argument('-n', type=int, help='Number of files to process (default: all)')
    batch_test_parser.add_argument('--output-dir', default='output_json', help='Directory containing AI JSON results')
    batch_test_parser.add_argument('--output-file', default='batch_test_results.json', help='Output JSON file for results')
    batch_test_parser.add_argument('--exclude-fields', nargs='+', help='Fields to exclude from evaluation')
    batch_test_parser.set_defaults(func=batch_test_command)

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
