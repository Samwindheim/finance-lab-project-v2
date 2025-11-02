"""
Run an evaluation of the accuracy of the extraction pipeline.
For a small sample: 
    python tests/run_evaluation.py --limit 5
For the full set: 
    python tests/run_evaluation.py
"""
import json
import os
import argparse
from decimal import Decimal, InvalidOperation
import re

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES_FILE = os.path.join(BASE_DIR, "tests", "sources.json")
MANUAL_INVESTORS_FILE = os.path.join(BASE_DIR, "tests", "manual_investors.json")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "output_json")
OUTPUT_REPORT_JSON = os.path.join(BASE_DIR, "accuracy_report.json")
OUTPUT_SUMMARY_MD = os.path.join(BASE_DIR, "accuracy_summary.md")
EXTRACTION_TYPE = "underwriters"

# --- ANSI Color Codes for Logging ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def log(color, message):
    print(f"{color}{message}{bcolors.ENDC}")

def normalize_name(name: str) -> str:
    """Normalizes investor names for reliable comparison."""
    if not name:
        return ""
    # Lowercase, remove common legal suffixes, and strip whitespace/punctuation
    name = name.lower()
    name = re.sub(r'\b(ab|ltd|inc|llc)\b', '', name)
    name = re.sub(r'[^a-z0-9]+', '', name)
    return name.strip()

def normalize_amount(amount) -> Decimal | None:
    """Normalizes amount to a Decimal for reliable comparison."""
    if amount is None:
        return None
    try:
        # Convert to string, remove spaces, then convert to Decimal
        return Decimal(str(amount).replace(" ", ""))
    except (InvalidOperation, TypeError):
        return None

def find_best_match(predicted_investor, potential_matches):
    """
    Finds the best match for a predicted investor from a list of potential ground truth candidates.
    The best match is the one with the fewest discrepancies. A perfect match has 0 errors.
    """
    best_match_candidate = None
    min_errors = float('inf')

    if not potential_matches:
        return None

    for gt_candidate in potential_matches:
        errors = compare_investor_data(predicted_investor, gt_candidate)
        num_errors = len(errors)

        if num_errors < min_errors:
            min_errors = num_errors
            best_match_candidate = gt_candidate
        
        # If we find a perfect match, we can stop searching immediately
        if num_errors == 0:
            return gt_candidate

    return best_match_candidate

def compare_investor_data(predicted, ground_truth):
    """Compares a single predicted investor to its ground truth match."""
    errors = []
    # Level comparison
    if predicted.get("level") != ground_truth.get("level"):
        errors.append(f"Level mismatch: Got {predicted.get('level')}, expected {ground_truth.get('level')}")

    # Amount comparison
    pred_amount = normalize_amount(predicted.get("amount_in_cash"))
    gt_amount = normalize_amount(ground_truth.get("amount_in_cash"))
    if pred_amount is not None and gt_amount is not None:
        if pred_amount != gt_amount:
            errors.append(f"Amount mismatch: Got {pred_amount}, expected {gt_amount}")
    elif pred_amount is not None and gt_amount is None:
        errors.append("Amount mismatch: Extracted an amount where none was expected")
    elif pred_amount is None and gt_amount is not None:
         errors.append(f"Amount mismatch: Did not extract an amount where one was expected ({gt_amount})")

    return errors

def main(limit: int | None):
    log(bcolors.HEADER, "=" * 70)
    log(bcolors.HEADER, "--- Starting Accuracy Evaluation ---")
    log(bcolors.HEADER, "=" * 70)

    # --- 1. Load Input Data ---
    log(bcolors.OKBLUE, "Loading sources and manual investor data...")
    with open(SOURCES_FILE, 'r') as f:
        sources = json.load(f)
    with open(MANUAL_INVESTORS_FILE, 'r') as f:
        manual_data = json.load(f)

    # Create a lookup map for ground truth data for efficient access
    # Add a check to ensure keys exist to prevent KeyErrors on malformed data
    ground_truth_map = {
        item["issue_id"]: item["investors"] 
        for item in manual_data 
        if "issue_id" in item and "investors" in item
    }
    
    # Apply limit if provided
    docs_to_process = sources[:limit] if limit else sources
    total_docs = len(docs_to_process)
    log(bcolors.OKBLUE, f"Processing {total_docs} documents.")

    # --- 2. Process Each Document ---
    report_data = []
    for i, source in enumerate(docs_to_process):
        pdf_filename = source.get("source_url")
        issue_id = source.get("issue_id")
        log(bcolors.HEADER, f"\n({i+1}/{total_docs}) Evaluating: {pdf_filename}")

        doc_report = {
            "document_name": pdf_filename,
            "issue_id": issue_id,
            "correct": 0, "incorrect": 0, "missing": 0, "false_positives": 0,
            "discrepancies": []
        }

        # --- Get Ground Truth ---
        ground_truth_investors = ground_truth_map.get(issue_id, [])
        if not ground_truth_investors:
            log(bcolors.WARNING, "  No manual data found for this issue_id. Skipping.")
            continue
            
        # --- Get Predicted Data ---
        pred_filename = f"{os.path.splitext(pdf_filename)[0]}_{EXTRACTION_TYPE}.json"
        pred_filepath = os.path.join(PREDICTIONS_DIR, pred_filename)
        if not os.path.exists(pred_filepath):
            log(bcolors.WARNING, "  Prediction JSON not found. Skipping.")
            continue
            
        with open(pred_filepath, 'r') as f:
            prediction_data = json.load(f)
        predicted_investors = prediction_data.get("investors", [])
        
        # --- Compare Predicted vs. Ground Truth ---
        gt_unmatched = list(ground_truth_investors) # A list to track who we haven't found yet
        
        for pred_investor in predicted_investors:
            # Find all potential matches by name in the currently available ground truth list
            normalized_pred_name = normalize_name(pred_investor.get("name"))
            potential_matches = [
                gt for gt in gt_unmatched 
                if normalize_name(gt.get("name")) == normalized_pred_name
            ]

            if not potential_matches:
                # This is a clear false positive, as no investor with this name was expected
                doc_report["false_positives"] += 1
                doc_report["discrepancies"].append({
                    "type": "False Positive",
                    "predicted": pred_investor,
                    "ground_truth": None,
                })
            else:
                # Find the best possible match from the candidates
                best_match = find_best_match(pred_investor, potential_matches)
                
                # Now that we've paired it, remove it from the list of available ground truth investors
                if best_match in gt_unmatched:
                    gt_unmatched.remove(best_match)
                
                errors = compare_investor_data(pred_investor, best_match)
                if not errors:
                    doc_report["correct"] += 1
                else:
                    doc_report["incorrect"] += 1
                    doc_report["discrepancies"].append({
                        "type": "Incorrect Value",
                        "predicted": pred_investor,
                        "ground_truth": best_match,
                        "errors": errors
                    })

        # Anything left in gt_unmatched is a missing investor
        doc_report["missing"] = len(gt_unmatched)
        for missing_investor in gt_unmatched:
            doc_report["discrepancies"].append({
                "type": "Missing",
                "predicted": None,
                "ground_truth": missing_investor
            })

        report_data.append(doc_report)
        log(bcolors.OKGREEN, f"  Evaluation complete. Correct: {doc_report['correct']}, Incorrect: {doc_report['incorrect']}, Missing: {doc_report['missing']}, False Positives: {doc_report['false_positives']}")

    # --- 3. Generate JSON Report ---
    log(bcolors.HEADER, "\n" + "=" * 70)
    log(bcolors.OKBLUE, f"Saving detailed JSON report to {OUTPUT_REPORT_JSON}...")
    with open(OUTPUT_REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    log(bcolors.OKGREEN, "Done.")

    # --- 4. Generate Markdown Summary ---
    log(bcolors.OKBLUE, f"Generating summary report at {OUTPUT_SUMMARY_MD}...")
    
    total_correct = sum(r['correct'] for r in report_data)
    total_incorrect = sum(r['incorrect'] for r in report_data)
    total_missing = sum(r['missing'] for r in report_data)
    total_false_positives = sum(r['false_positives'] for r in report_data)
    total_extracted = total_correct + total_incorrect + total_false_positives
    docs_100_accuracy = sum(1 for r in report_data if r['incorrect'] == 0 and r['missing'] == 0 and r['false_positives'] == 0)

    summary = f"""# Extraction Accuracy Summary

## Overall Metrics
- **Total Documents Processed:** {len(report_data)}
- **Total Investors Extracted (Predicted):** {total_extracted}
- **Correct Datapoint Extractions:** {total_correct}
- **Incorrect Datapoint Extractions:** {total_incorrect}
- **Missing Datapoint Extractions:** {total_missing}
- **False Positives:** {total_false_positives}
- **Documents with 100% Accuracy:** {docs_100_accuracy}
- **Documents with < 100% Accuracy:** {len(report_data) - docs_100_accuracy}

## Discrepancy Details
"""
    for report in report_data:
        if report['discrepancies']:
            summary += f"\n### 📄 {report['document_name']}\n"
            for disc in report['discrepancies']:
                summary += f"- **Type:** {disc['type']}\n"
                if disc['type'] == 'Missing':
                    summary += f"  - **Expected:** `{disc['ground_truth'].get('name')}`\n"
                elif disc['type'] == 'False Positive':
                    summary += f"  - **Extracted:** `{disc['predicted'].get('name')}`\n"
                elif disc['type'] == 'Incorrect Value':
                    summary += f"  - **Investor:** `{disc['predicted'].get('name')}`\n"
                    summary += f"  - **Errors:** {', '.join(disc['errors'])}\n"

    with open(OUTPUT_SUMMARY_MD, 'w', encoding='utf-8') as f:
        f.write(summary)
    log(bcolors.OKGREEN, "Summary report saved.")
    log(bcolors.HEADER, "=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the accuracy of the investor extraction pipeline.")
    parser.add_argument(
        '-l', '--limit',
        type=int,
        help="Limit the evaluation to the first N documents from sources.json."
    )
    args = parser.parse_args()
    main(args.limit)
