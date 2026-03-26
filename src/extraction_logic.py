"""
Core Data Extraction and Processing Logic.

This module contains the primary functions for performing the Retrieval-Augmented Generation (RAG)
extraction from various document types. It encapsulates the logic for querying the vector index,
selecting relevant pages, calling the vision/language models, and post-processing the results.
"""
import os
import json
from typing import List, Dict, Union, get_args, get_origin

from pdf_indexer import PDFIndexer
from llm import get_json_from_image, get_json_from_text
from html_processor import extract_text_from_html
import config
from utils import find_document_info, clean_and_parse_json, get_db
from models import ExtractionResult, Investor, ImportantDates, FinalOutput, DocumentEntry, DocumentClassification
from logger import setup_logger

logger = setup_logger(__name__)


def _db_save_enabled() -> bool:
    """Controls whether AI extraction rows are persisted to the database."""
    return os.getenv("ENABLE_DB_SAVE", "true").lower() == "true"


def classify_html_document(url: str) -> DocumentClassification:
    """
    Fetches the first ~3000 chars of an HTML document and asks the LLM to
    classify its source_type and issue_type. Falls back to None values on failure.
    Only pass the first 3000 characters of the HTML to the LLM.
    """
    import requests
    text = ""
    # print input text to classification_input.txt
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        # Pass the raw HTML directly to the processor (not as a path/URL)
        text = extract_text_from_html(resp.text, raw_html=True)
        with open("classification_input.txt", "w") as f:
            f.write(text)
    except Exception as e:
        logger.warning(f"Could not fetch HTML for classification: {e}")
        
    if not text:
        return DocumentClassification()


    raw = get_json_from_text(text, extraction_type="classify_document")
    parsed = clean_and_parse_json(raw)
    if not parsed:
        return DocumentClassification()

    try:
        return DocumentClassification(**parsed)
    except Exception:
        return DocumentClassification()


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


def post_process_and_save(parsed_json: dict, source_path: str, extraction_field: str, source_pages: list, output_path: str, issue_id: str = None, source_url: str = None):
    """Post-processes and saves the extracted JSON data to a specified file path."""
    doc_info = {}
    try:
        db = get_db()
        if getattr(db, "engine", None):
            doc_info = find_document_info(source_path, issue_id=issue_id)
    except Exception:
        doc_info = {}

    if not issue_id:
        issue_id = doc_info.get("issue_id")
    doc_id = doc_info.get("doc_id")
    # Prefer DB value, then caller-supplied URL, then local path as last resort
    source_url = doc_info.get("source_url") or source_url or source_path
    
    # The parsed_json from the model should contain the extraction_field as a key
    field_value = parsed_json.get(extraction_field)
    
    final_output = {
        "issue_id": issue_id,
        "doc_id": doc_id,
        "source_url": source_url,
        "source_document": os.path.basename(source_path),
        "source_pages": source_pages,
        extraction_field: field_value
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    return output_path


def _validate_and_save(json_data: str, source_path: str, extraction_field: str, source_pages: list, output_path: str, issue_id: str = None, source_url: str = None) -> str | None:
    """Parses, validates, and saves LLM output. Returns the saved path on success, None on failure."""
    parsed_json = clean_and_parse_json(json_data)
    if not parsed_json:
        return None
    try:
        ExtractionResult.model_validate(parsed_json)
        return post_process_and_save(parsed_json, source_path, extraction_field, source_pages, output_path, issue_id=issue_id, source_url=source_url)
    except Exception as e:
        logger.warning(f"LLM output for {extraction_field} failed validation: {e}")
        post_process_and_save(parsed_json, source_path, extraction_field, source_pages, output_path, issue_id=issue_id, source_url=source_url)
        return None


def extract_from_pdf(pdf_path: str, search_query: str, extraction_prompt: str, extraction_field: str, output_path: str, page_selection_strategy: str = "consecutive", issue_id: str = None, source_url: str = None):
    """Performs a full RAG extraction on a single PDF document."""

    index_name = os.path.splitext(os.path.basename(pdf_path))[0]
    indexer = PDFIndexer(index_path=os.path.join(config.FAISS_INDEX_DIR, index_name))
    if indexer.index.ntotal == 0:
        logger.info(f"Index not found for {os.path.basename(pdf_path)}. Building it now...")
        indexer.index_pdf(pdf_path)

    # 1. Query the index
    logger.info(f"Querying for relevant pages (strategy: {page_selection_strategy})...")
    results = indexer.query(search_query, top_k=6)
    if not results:
        logger.warning(f"Could not find any relevant pages for field '{extraction_field}' in {os.path.basename(pdf_path)}.")
        return None

    # 2. Select pages
    if page_selection_strategy == "top_hit":
        page_numbers = [results[0]['page_number']]
    else:
        page_numbers = select_consecutive_pages(results)
    logger.info(f"Selected {len(page_numbers)} pages to analyze: {page_numbers}")

    # 3. Extract images and text (single PDF open)
    saved_image_paths, page_texts = indexer.extract_pages(pdf_path, page_numbers)

    if not saved_image_paths:
        logger.error(f"Could not extract any images for the identified pages in {os.path.basename(pdf_path)}.")
        return None

    # 4. Call LLM and save
    combined_text = "\n\n--- Page Separator ---\n\n".join(page_texts)
    json_data = get_json_from_image(saved_image_paths, combined_text, prompt_text=extraction_prompt, extraction_type=extraction_field)
    if json_data:
        return _validate_and_save(json_data, pdf_path, extraction_field, page_numbers, output_path, issue_id=issue_id, source_url=source_url)
    return None


def extract_from_html(html_path: str, extraction_prompt: str, extraction_field: str, output_path: str, issue_id: str = None):
    """Performs a full-text extraction on a single HTML document."""

    text = extract_text_from_html(html_path, preserve_tables=True)
    if not text:
        logger.error(f"Could not extract any text from the HTML file: {html_path}")
        return None

    json_data = get_json_from_text(text, prompt_text=extraction_prompt, extraction_type=extraction_field)
    if json_data:
        return _validate_and_save(json_data, html_path, extraction_field, [1], output_path, issue_id=issue_id)
    return None


def merge_and_finalize_outputs(issue_id: str, extraction_field: str, temp_files: List[str], final_output_path: str, warnings: List[str] = None):
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
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                temp_result = json.load(f)
            
            doc_name = temp_result.get("source_document")
            doc_id = temp_result.get("doc_id")
            source_url = temp_result.get("source_url")
            if not doc_name:
                continue

            # Get the extracted field value and source pages
            field_value = temp_result.get(extraction_field)
            source_pages = temp_result.get("source_pages", [])
            if field_value is None:
                continue

            # Save to AI extractions database table (toggleable for Lambda MVP)
            if _db_save_enabled():
                try:
                    db_data = {extraction_field: field_value, "source_pages": source_pages}
                    get_db().save_ai_extraction(
                        issue_id=issue_id,
                        doc_id=doc_id,
                        extraction_field=extraction_field,
                        data=db_data,
                        source_url=source_url,
                        warnings=warnings or []
                    )
                    logger.info(f"Saved {extraction_field} to database.")
                except Exception as e:
                    logger.error(f"Failed to save AI extraction to database: {e}")
            else:
                logger.info("DB save disabled (ENABLE_DB_SAVE=false).")

            # Ensure document entry exists
            if doc_name not in all_data:
                all_data[doc_name] = DocumentEntry(issue_id=issue_id, id=doc_id)

            doc_entry = all_data[doc_name]
            
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
                doc_entry.investors_source_pages = source_pages
            
            elif field_info:
                try:
                    # Determine the type for validation
                    annotation = field_info.annotation
                    if get_origin(annotation) is Union:
                        # Extract the non-None type from Union/Optional
                        annotation = next((a for a in get_args(annotation) if a is not type(None)), annotation)
                    
                    if hasattr(annotation, "model_validate"):
                        validated_value = annotation.model_validate(field_value)
                        # Set source pages on the validated model
                        if hasattr(validated_value, "source_pages"):
                            validated_value.source_pages = source_pages
                        setattr(doc_entry, extraction_field, validated_value)
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
                    # Update source pages for important_dates if it's being updated
                    doc_entry.important_dates.source_pages = source_pages
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
            
            # Special handling for investors to keep source_pages next to it
            if field_name == "investors":
                if doc_entry.investors is not None:
                    ordered_doc["investors"] = [i.model_dump(exclude_none=True) for i in doc_entry.investors]
                    if doc_entry.investors_source_pages is not None:
                        ordered_doc["investors_source_pages"] = doc_entry.investors_source_pages
                continue
            
            if field_name == "investors_source_pages":
                continue # Handled above
            
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
        
    logger.info(f"Updated results locally to: {final_output_path}")

    if os.getenv("ENABLE_RESULT_LOG", "false").lower() == "true":
        logger.info("--- EXTRACTION RESULT ---")
        logger.info(json.dumps(sorted_output, indent=2, ensure_ascii=False))
        logger.info("--- END RESULT ---")
