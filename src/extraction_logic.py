"""
Core Data Extraction and Processing Logic.

This module contains the primary functions for performing the Retrieval-Augmented Generation (RAG)
extraction from various document types. It encapsulates the logic for querying the vector index,
selecting relevant pages, calling the vision/language models, and post-processing the results.

These functions are designed to be reusable and are orchestrated by higher-level scripts
like `run_extraction.py`.
"""
import os
import json
from typing import List, Dict, Union, get_args, get_origin

from pdf_indexer import PDFIndexer
from llm import get_json_from_image, get_json_from_text
from html_processor import extract_text_from_html
import config
from utils import find_issue_id, find_document_info, clean_and_parse_json
from models import ExtractionResult, Investor, ImportantDates, OfferingTerms, OfferingOutcome, GeneralInfo, FinalOutput, DocumentEntry
from logger import setup_logger

logger = setup_logger(__name__)

# --- Load sources data once for performance ---
# This data is needed for `find_issue_id` in the post-processing step.
SOURCES_DATA = None
HTML_SOURCES_DATA = None

def _load_sources_data():
    """Lazy loader for the sources data to avoid loading it if not needed."""
    global SOURCES_DATA, HTML_SOURCES_DATA
    if SOURCES_DATA is None:
        from utils import load_json_file
        SOURCES_DATA = load_json_file(config.PDF_SOURCES_FILE)
    if HTML_SOURCES_DATA is None:
        from utils import load_json_file
        HTML_SOURCES_DATA = load_json_file(config.HTML_SOURCES_FILE)


def select_consecutive_pages(results: List[Dict], max_pages: int = 4) -> List[int]:
    """
    Selects a consecutive block of pages around the top search results from a PDF.
    This logic is crucial for capturing tables that span multiple pages.
    
    Args:
        results: List of search results with page numbers and distances
        max_pages: Maximum number of pages to select (default: 4)
    """
    if not results:
        return []

    SIMILARITY_DISTANCE_THRESHOLD = 1.05 # 5% threshold

    # Start with the top result and include the second result if its score is close.
    top_result = results[0]
    pages_to_extract = [top_result]
    
    # Check if second result is close enough to include
    if len(results) > 1 and len(pages_to_extract) < max_pages:
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
    if len(pages_to_extract) < max_pages:
        if current_page in results_by_page:
            pages_to_extract.append(results_by_page[current_page])
        else:
            # Synthetic page to ensure the next page is included even if not in top results
            pages_to_extract.append({'page_number': current_page, 'distance': float('inf'), 'text': ''})
    
    current_page += 1
    while current_page in results_by_page and len(pages_to_extract) < max_pages:
        pages_to_extract.append(results_by_page[current_page])
        current_page += 1

    # Look backward to find consecutive pages that are also in the search results.
    current_page = top_result['page_number'] - 1
    while current_page > 0 and current_page in results_by_page and len(pages_to_extract) < max_pages:
        pages_to_extract.append(results_by_page[current_page])
        current_page -= 1

    # Remove duplicates by page number, keeping the first occurrence
    unique_pages = []
    seen_page_numbers = set()
    for page in pages_to_extract:
        if page['page_number'] not in seen_page_numbers:
            unique_pages.append(page)
            seen_page_numbers.add(page['page_number'])
    
    # Sort the collected pages by page number to ensure correct order
    unique_pages.sort(key=lambda p: p['page_number'])

    return [p['page_number'] for p in unique_pages]


def post_process_and_save(parsed_json: dict, source_path: str, extraction_field: str, source_pages: list, output_path: str):
    """Post-processes and saves the extracted JSON data to a specified file path."""
    _load_sources_data() # Ensure source data is loaded
    
    doc_info = find_document_info(source_path, SOURCES_DATA, HTML_SOURCES_DATA)
    issue_id = doc_info.get("issue_id")
    doc_id = doc_info.get("doc_id")
    
    if not issue_id:
        logger.warning(f"Could not find issue_id for '{os.path.basename(source_path)}'.")

    # The parsed_json from the model should contain the extraction_field as a key
    field_value = parsed_json.get(extraction_field)
    
    final_output = {
        "issue_id": issue_id,
        "doc_id": doc_id,
        "source_document": os.path.basename(source_path),
        "source_pages": source_pages,
        extraction_field: field_value
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    logger.info(f"Successfully extracted and saved data to: {output_path}")
    return output_path


def extract_from_pdf(pdf_path: str, search_query: str, extraction_prompt: str, extraction_field: str, output_path: str, page_selection_strategy: str = "consecutive"):
    """
    Reusable logic to perform a full RAG extraction on a single PDF document.
    """
    logger.info(f"Processing PDF: {os.path.basename(pdf_path)}")
    
    # Simplified index path generation
    index_name = os.path.splitext(os.path.basename(pdf_path))[0]
    index_path = os.path.join(config.FAISS_INDEX_DIR, index_name)

    indexer = PDFIndexer(index_path=index_path)
    if indexer.index.ntotal == 0:
        logger.info(f"Index not found for {os.path.basename(pdf_path)}. Building it now...")
        indexer.index_pdf(pdf_path)

    # 1. Query the index
    logger.info(f"Querying for relevant pages (strategy: {page_selection_strategy})...")
    results = indexer.query(search_query, top_k=5)

    if not results:
        logger.warning(f"Could not find any relevant pages for field '{extraction_field}' in {os.path.basename(pdf_path)}.")
        return None

    # 2. Select pages based on the specified strategy
    page_numbers = []
    if page_selection_strategy == "top_hit":
        # For simple fields, just take the page from the single best result.
        page_numbers = [results[0]['page_number']]
    else: # Default to "consecutive"
        page_numbers = select_consecutive_pages(results)
        
    logger.info(f"Selected {len(page_numbers)} pages to analyze: {page_numbers}")

    # 3. Extract images and text for context
    saved_image_paths = []
    page_texts = []
    for page_num in page_numbers:
        # Check if the page exists in the document before trying to extract
        if page_num > 0 and page_num <= indexer.get_page_count(pdf_path):
            img_path = indexer.extract_page_as_image(pdf_path, page_num)
            if img_path:
                saved_image_paths.append(img_path)
            
            text = indexer.get_text_for_page(pdf_path, page_num)
            if text:
                page_texts.append(text)
    
    if not saved_image_paths:
        logger.error(f"Could not extract any images for the identified pages in {os.path.basename(pdf_path)}.")
        return None

    # 4. Call LLM and save result
    combined_text = "\n\n--- Page Separator ---\n\n".join(page_texts)
    json_data = get_json_from_image(saved_image_paths, combined_text, prompt_text=extraction_prompt, extraction_type=extraction_field)
    if json_data:
        parsed_json = clean_and_parse_json(json_data)
        if parsed_json:
            try:
                # Validate the LLM output against our model
                ExtractionResult.model_validate(parsed_json)
                return post_process_and_save(parsed_json, pdf_path, extraction_field, page_numbers, output_path)
            except Exception as e:
                logger.warning(f"LLM output for {extraction_field} failed validation: {e}")
                # We'll still save it for debugging, but return None to indicate failure
                post_process_and_save(parsed_json, pdf_path, extraction_field, page_numbers, output_path)
    return None


def extract_from_html(html_path: str, extraction_prompt: str, extraction_field: str, output_path: str):
    """
    Reusable logic to perform a full-text extraction on a single HTML document.
    """
    logger.info(f"Processing HTML: {os.path.basename(html_path)}")
    
    # 1. Extract text
    text = extract_text_from_html(html_path, preserve_tables=True)
    if not text:
        logger.error(f"Could not extract any text from the HTML file: {html_path}")
        return None

    # 2. Call LLM and save result
    json_data = get_json_from_text(text, prompt_text=extraction_prompt, extraction_type=extraction_field)
    if json_data:
        parsed_json = clean_and_parse_json(json_data)
        if parsed_json:
            try:
                # Validate the LLM output against our model
                ExtractionResult.model_validate(parsed_json)
                # For HTML, we consider it as a single "page"
                return post_process_and_save(parsed_json, html_path, extraction_field, [1], output_path)
            except Exception as e:
                logger.warning(f"LLM output for {extraction_field} failed validation: {e}")
                post_process_and_save(parsed_json, html_path, extraction_field, [1], output_path)
    return None


def merge_and_finalize_outputs(issue_id: str, extraction_field: str, temp_files: List[str], final_output_path: str):
    """
    Merges data from multiple temporary files and saves them to a final output file
    grouped by document name using Pydantic models for validation.
    """
    if not temp_files:
        logger.warning(f"No temporary files to merge for field '{extraction_field}'.")
        return

    # 1. Load existing data if it exists
    all_data = {}
    if os.path.exists(final_output_path):
        try:
            with open(final_output_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                # Validate existing data
                all_data = FinalOutput.model_validate(raw_data).root
        except Exception as e:
            logger.warning(f"Could not parse or validate existing output file: {e}. Creating new.")
            all_data = {}

    # 2. Process each temporary result file
    for file_path in temp_files:
        # ... existing logic for processing temp files ...
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                temp_result = json.load(f)
            
            doc_name = temp_result.get("source_document")
            if not doc_name:
                continue

            # Get the extracted field value
            field_value = temp_result.get(extraction_field)
            if field_value is None:
                continue

            # Ensure document entry exists
            if doc_name not in all_data:
                all_data[doc_name] = DocumentEntry(issue_id=issue_id, id=temp_result.get("doc_id"))
            
            doc_entry = all_data[doc_name]
            
            # Update doc_id if it's missing but present in temp_result
            if not doc_entry.id and temp_result.get("doc_id"):
                doc_entry.id = temp_result.get("doc_id")
            
            # --- Dynamic field handling based on DocumentEntry model ---
            field_info = DocumentEntry.model_fields.get(extraction_field)
            
            if extraction_field == "investors":
                # Special logic for investors: Deduplicate and format numbers
                investors = []
                seen_entries = set()
                for item in field_value:
                    try:
                        investor = Investor.model_validate(item)
                        if investor.name and (investor.name, investor.level) not in seen_entries:
                            # Format numbers to strings as requested previously
                            if isinstance(investor.amount_in_cash, (int, float)):
                                investor.amount_in_cash = f"{investor.amount_in_cash:.3f}"
                            if isinstance(investor.amount_in_percentage, (int, float)):
                                investor.amount_in_percentage = str(investor.amount_in_percentage)
                            
                            investors.append(investor)
                            seen_entries.add((investor.name, investor.level))
                    except Exception as e:
                        logger.warning(f"Skipping invalid investor entry in {doc_name}: {e}")
                
                doc_entry.investors = investors
            
            elif field_info:
                try:
                    # Determine the type for validation
                    annotation = field_info.annotation
                    if get_origin(annotation) is Union:
                        # Extract the non-None type from Union/Optional
                        annotation = next((a for a in get_args(annotation) if a is not type(None)), annotation)
                    
                    if hasattr(annotation, "model_validate"):
                        setattr(doc_entry, extraction_field, annotation.model_validate(field_value))
                    else:
                        setattr(doc_entry, extraction_field, field_value)
                except Exception as e:
                    logger.warning(f"Skipping invalid {extraction_field} entry in {doc_name}: {e}")
            
            else:
                # Field not explicitly in DocumentEntry, but could be an individual field for important_dates
                if extraction_field in ImportantDates.model_fields:
                    if not doc_entry.important_dates:
                        doc_entry.important_dates = ImportantDates()
                    setattr(doc_entry.important_dates, extraction_field, field_value)
                else:
                    # Truly extra field (allowed by DocumentEntry Config)
                    setattr(doc_entry, extraction_field, field_value)

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Could not read temp file {file_path}: {e}")
            continue

    # 3. Save the finalized structure (ordered by DocumentEntry field definitions)
    sorted_output = {}
    for doc_name in sorted(all_data.keys()):
        doc_entry = all_data[doc_name]
        
        # Start with the basic issue info
        ordered_doc = {
            "issue_id": doc_entry.issue_id,
            "id": doc_entry.id
        }
        
        # Dynamically add all other fields from DocumentEntry if they have data
        for field_name in DocumentEntry.model_fields.keys():
            if field_name in ["issue_id", "id"]:
                continue
            
            value = getattr(doc_entry, field_name)
            if value is not None:
                if hasattr(value, "model_dump"):
                    ordered_doc[field_name] = value.model_dump(exclude_none=True)
                elif isinstance(value, list) and all(hasattr(i, "model_dump") for i in value):
                    ordered_doc[field_name] = [i.model_dump(exclude_none=True) for i in value]
                else:
                    ordered_doc[field_name] = value
        
        # Add any other extra fields not defined in DocumentEntry
        doc_dict = doc_entry.model_dump(
            exclude=set(DocumentEntry.model_fields.keys()),
            exclude_none=True
        )
        for key in sorted(doc_dict.keys()):
            ordered_doc[key] = doc_dict[key]
            
        sorted_output[doc_name] = ordered_doc

    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_output, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Successfully updated results in: {final_output_path}")
