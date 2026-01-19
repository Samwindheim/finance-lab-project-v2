import os
import json
import argparse
import pandas as pd
from src.database import FinanceDB
from tests.evaluator import Evaluator
from tabulate import tabulate
from src.models import get_fields_for_category

def log_header(text):
    print(f"\n{'='*20} {text} {'='*20}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate AI extraction against SQL Database Ground Truth")
    parser.add_argument("--issue-id", required=True, help="The issue_id to evaluate")
    parser.add_argument("--output-dir", default="output_json", help="Directory containing AI JSON results")
    args = parser.parse_args()

    db = FinanceDB()
    evaluator = Evaluator()
    
    issue_id = args.issue_id
    ai_file = os.path.join(args.output_dir, f"{issue_id}_extraction.json")

    if not os.path.exists(ai_file):
        print(f"Error: AI extraction file not found: {ai_file}")
        return

    # 1. Load AI Data
    with open(ai_file, 'r') as f:
        ai_data_full = json.load(f)
    
    # AI data is organized by document key, we merge them for evaluation
    merged_ai_data = {}
    for doc_key, data in ai_data_full.items():
        for field, value in data.items():
            if isinstance(value, dict):
                merged_ai_data.setdefault(field, {}).update(value)
            elif isinstance(value, list):
                merged_ai_data.setdefault(field, []).extend(value)
            else:
                merged_ai_data[field] = value

    # 2. Load DB Ground Truth
    db_issue = db.get_issue_data(issue_id)
    db_investors = db.get_investors_data(issue_id)

    if not db_issue:
        print(f"Error: No data found in 'issues' table for ID: {issue_id}")
        return

    # --- EVALUATE INVESTORS ---
    log_header("INVESTOR EVALUATION")
    ai_investors = merged_ai_data.get("investors", [])
    gt_investors = db_investors.to_dict('records')
    
    inv_report = evaluator.compare_investors(ai_investors, gt_investors)
    
    print(f"Summary: {inv_report['correct']} Correct, {inv_report['incorrect']} Incorrect, "
          f"{inv_report['missing']} Missing, {inv_report['false_positives']} False Positives")

    if inv_report['details']:
        table_data = []
        # Sort details so Correct matches are together, etc.
        sorted_details = sorted(inv_report['details'], key=lambda x: x['type'])
        for d in sorted_details:
            status_icon = "✅" if d['type'] == "Correct" else "❌"
            row = [
                f"{status_icon} {d['type']}",
                d['predicted'].get('name') if d['predicted'] else "N/A",
                d['ground_truth'].get('name') if d['ground_truth'] else "N/A",
                ", ".join(d.get('errors', [])) if 'errors' in d else ""
            ]
            table_data.append(row)
        print(tabulate(table_data, headers=["Status/Type", "AI Found", "DB Ground Truth", "Errors"], tablefmt="grid"))

    # --- EVALUATE TERMS & DATES ---
    log_header("TERMS & DATES EVALUATION")
    
    # Map model fields to DB columns
    fields_to_check = {
        "important_dates": get_fields_for_category("important_dates"),
        "offering_terms": get_fields_for_category("offering_terms"),
        "offering_outcome": get_fields_for_category("offering_outcome")
    }

    all_field_details = []
    for model_group, fields in fields_to_check.items():
        ai_group_data = merged_ai_data.get(model_group, {})
        field_results = evaluator.compare_fields(ai_group_data, db_issue, fields)
        
        for r in field_results:
            status_icon = "✅" if r['is_match'] else "❌"
            all_field_details.append([
                status_icon,
                model_group,
                r['field'],
                r['predicted'] if r['predicted'] is not None else "-",
                r['ground_truth'] if r['ground_truth'] is not None else "-"
            ])

    if all_field_details:
        print(tabulate(all_field_details, headers=["Match", "Group", "Field", "AI Value", "DB Value"], tablefmt="grid"))
    else:
        print("No fields found to evaluate.")

if __name__ == "__main__":
    main()
