"""
Run a batch extraction on the PDFs listed in sources.json to generate output JSON files for accuracy testing.
For a small sample: 
    python tests/run_batch_extraction.py --limit 5
For the full set: 
    python tests/run_batch_extraction.py
"""
import json
import os
import subprocess
import requests
import argparse
from urllib.parse import urlparse
import sys

# --- Configuration ---
# Using Path objects for OS-independent path handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES_FILE = os.path.join(BASE_DIR, "tests", "sources.json")
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
OUTPUT_JSON_DIR = os.path.join(BASE_DIR, "output_json")
EXTRACTION_TYPE = "underwriters"

# --- ANSI Color Codes for Logging ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log(color, message):
    """Prints a colored message to the console."""
    print(f"{color}{message}{bcolors.ENDC}")

def run_pipeline_command(command_args: list) -> bool:
    """Runs a pipeline command (e.g., index, extract) and returns True on success."""
    base_command = [os.path.join(BASE_DIR, "run.sh")]
    full_command = base_command + command_args
    
    command_str = ' '.join(full_command)
    log(bcolors.OKBLUE, f"  Running command: {command_str}")
    
    try:
        process = subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        # Stream output in real-time for better feedback
        for line in iter(process.stdout.readline, ''):
            print(f"    > {line.strip()}")
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            log(bcolors.FAIL, f"  Command '{command_str}' failed with exit code {return_code}.")
            return False
        
        return True
    except Exception as e:
        log(bcolors.FAIL, f"  An exception occurred while running '{command_str}': {e}")
        return False

def download_pdf(url: str, dest_folder: str) -> str | None:
    """Downloads a PDF from a URL if it doesn't already exist."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filename = os.path.basename(urlparse(url).path)
    dest_path = os.path.join(dest_folder, filename)

    if os.path.exists(dest_path):
        return dest_path
    
    log(bcolors.OKBLUE, f"  Downloading PDF: {url}")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return dest_path
    except requests.exceptions.RequestException as e:
        log(bcolors.FAIL, f"  Failed to download {url}. Error: {e}")
        return None

def main(limit: int | None):
    """Main function to run the batch extraction process."""
    log(bcolors.HEADER, "=" * 70)
    log(bcolors.HEADER, "--- Starting Batch Extraction for Accuracy Measurement ---")
    log(bcolors.HEADER, "=" * 70)

    # --- 1. Load Sources ---
    if not os.path.exists(SOURCES_FILE):
        log(bcolors.FAIL, f"FATAL: Source file not found at '{SOURCES_FILE}'")
        sys.exit(1)

    with open(SOURCES_FILE, 'r') as f:
        sources = json.load(f)
    
    # Apply the limit if one was provided
    if limit is not None and limit > 0:
        log(bcolors.WARNING, f"Limiting run to the first {limit} document(s).")
        sources_to_process = sources[:limit]
    else:
        sources_to_process = sources
    
    total_docs = len(sources_to_process)
    log(bcolors.OKBLUE, f"Found {len(sources)} total sources. Processing {total_docs}.")

    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    # --- 2. Process each source document ---
    for i, source in enumerate(sources_to_process):
        pdf_url = source.get("full_url")
        if not pdf_url:
            log(bcolors.WARNING, f"Skipping source at index {i} due to missing 'full_url'.")
            continue
            
        pdf_filename = os.path.basename(urlparse(pdf_url).path)
        log(bcolors.HEADER, f"\n({i+1}/{total_docs}) Processing: {pdf_filename}")

        # --- Step 2a: Check for existing output ---
        expected_output_filename = f"{os.path.splitext(pdf_filename)[0]}_{EXTRACTION_TYPE}.json"
        output_path = os.path.join(OUTPUT_JSON_DIR, expected_output_filename)

        if os.path.exists(output_path):
            log(bcolors.OKGREEN, "  Output JSON already exists. Skipping.")
            skipped_count += 1
            success_count += 1 # Count skipped as a success for the purpose of the run
            continue

        # --- Step 2b: Download PDF ---
        pdf_path = download_pdf(pdf_url, PDF_DIR)
        if not pdf_path:
            log(bcolors.FAIL, "  Download failed. Moving to next document.")
            failed_count += 1
            continue

        # --- Step 2c: Index PDF to ensure it exists ---
        log(bcolors.OKBLUE, "  Step 1: Indexing PDF...")
        index_command_args = ["index", pdf_path]
        if not run_pipeline_command(index_command_args):
            log(bcolors.FAIL, "  Indexing process failed. Moving to next document.")
            failed_count += 1
            continue

        # --- Step 2d: Run Extraction ---
        log(bcolors.OKBLUE, "  Step 2: Extracting data...")
        extract_command_args = ["extract", EXTRACTION_TYPE, pdf_path]
        if run_pipeline_command(extract_command_args):
            log(bcolors.OKGREEN, "  Successfully extracted and saved data.")
            success_count += 1
        else:
            log(bcolors.FAIL, "  Extraction process failed.")
            failed_count += 1

    # --- 3. Final Report ---
    log(bcolors.HEADER, "\n" + "=" * 70)
    log(bcolors.HEADER, "--- Batch Extraction Complete ---")
    log(bcolors.HEADER, f"Total Documents Processed: {total_docs}")
    log(bcolors.OKGREEN,   f"Successfully Extracted:    {success_count - skipped_count}")
    log(bcolors.OKBLUE,    f"Skipped (already done):    {skipped_count}")
    log(bcolors.FAIL,      f"Failed:                    {failed_count}")
    log(bcolors.HEADER, "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch extraction on PDFs listed in sources.json to generate prediction files for accuracy testing."
    )
    parser.add_argument(
        '-l', '--limit',
        type=int,
        default=None,
        help="Limit the run to the first N documents in the sources file. Useful for testing."
    )
    args = parser.parse_args()
    main(args.limit)
