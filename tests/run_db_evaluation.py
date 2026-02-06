import os
import json
import argparse
import pandas as pd
from decimal import Decimal
from src.database import FinanceDB
from tests.evaluator import Evaluator
from tabulate import tabulate
from src.models import get_fields_for_category, ExtractionResult

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

    # 2. Detect Conflicts Across Documents
    conflicts = evaluator.detect_conflicts(ai_data_full)
    if conflicts:
        log_header("MULTI-DOCUMENT CONFLICTS DETECTED")
        conflict_table = []
        for category, fields in conflicts.items():
            for field, instances in fields.items():
                for inst in instances:
                    conflict_table.append([
                        category, 
                        field, 
                        inst['doc'][:30] + "..." if len(inst['doc']) > 30 else inst['doc'], 
                        inst['val']
                    ])
        print(tabulate(conflict_table, headers=["Category", "Field", "Source Document", "Conflicting Value"], tablefmt="grid"))
        print("\nNOTE: The evaluation below uses the 'latest' value seen for each field.")

    # 3. Load DB Ground Truth
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
            name = d['predicted'].get('name', '') if d['predicted'] else ''
            level = d['predicted'].get('level') if d['predicted'] else None
            norm_name = evaluator.normalize_name(name)
            
            # Check for conflict using name and level
            inv_key = f"{norm_name}.{level}"
            has_conflict = "investors" in conflicts and inv_key in conflicts["investors"]
            
            # Skip matches that have no conflicts to keep output focused on issues
            if d['type'] == "Correct" and not has_conflict:
                continue

            status_icon = "⚠️" if has_conflict else ("✅" if d['type'] == "Correct" else "❌")
            row = [
                f"{status_icon} {d['type']}",
                d['predicted'].get('name') if d['predicted'] else "N/A",
                d['ground_truth'].get('name') if d['ground_truth'] else "N/A",
                ", ".join(d.get('errors', [])) if 'errors' in d else ""
            ]
            table_data.append(row)
        print(tabulate(table_data, headers=["Status/Type", "AI Found", "DB Ground Truth", "Errors"], tablefmt="grid"))

    # --- EVALUATE TERMS, OUTCOME & DATES ---
    log_header("TERMS, OUTCOME & DATES EVALUATION")
    
    # Format values for display to avoid scientific notation
    def format_val(val):
        if val is None or str(val).strip() == "":
            return "-"
        try:
            # If it's a number, format without scientific notation
            # First convert to string and remove any existing formatting
            s_val = str(val).replace(" ", "").replace(",", "")
            num = Decimal(s_val)
            if num == num.to_integral_value():
                return f"{num:,.0f}".replace(",", " ") # Use space as thousands separator
            # For percentages/decimals, keep up to 2 decimal places if needed, but no scientific notation
            return f"{num:f}".rstrip('0').rstrip('.')
        except:
            return str(val)

    # Dynamically build fields to check from ExtractionResult model
    # Exclude investors and general_info and isin_units and isin_rights as requested
    fields_to_check = {
        field_name: get_fields_for_category(field_name)
        for field_name in ExtractionResult.model_fields.keys()
        if field_name not in ["investors"]
    }

    all_field_details = []
    terms_correct = 0
    terms_share_matches = 0
    terms_incorrect = 0
    terms_missing = 0
    terms_conflicts = 0
    terms_manual_check = 0

    for model_group, fields in fields_to_check.items():
        ai_group_data = merged_ai_data.get(model_group, {})
        
        if not isinstance(ai_group_data, dict):
            ai_eval_data = {model_group: ai_group_data}
        else:
            ai_eval_data = ai_group_data

        field_results = evaluator.compare_fields(ai_eval_data, db_issue, fields)
        
        for r in field_results:
            has_conflict = model_group in conflicts and r['field'] in conflicts[model_group]
            
            # Handle manual check flag (mutually exclusive pairs)
            if r.get('needs_manual_check'):
                terms_manual_check += 1
                status_icon = "🔍"  # Magnifying glass icon for manual check
                
                all_field_details.append([
                    status_icon,
                    model_group,
                    r['field'],
                    format_val(r['predicted']),
                    f"NULL (Manually check)"
                ])
                continue
            
            if r['is_match'] and not has_conflict:
                if r.get('is_share_match'):
                    terms_share_matches += 1
                else:
                    terms_correct += 1
                continue

            # Determine if it's Missing or Incorrect
            # Missing: DB has a value, but AI does not.
            is_missing = (r['predicted'] is None or str(r['predicted']).strip() == "") and \
                         (r['ground_truth'] is not None and str(r['ground_truth']).strip() != "")

            if has_conflict:
                terms_conflicts += 1
                status_icon = "⚠️"
            elif is_missing:
                terms_missing += 1
                status_icon = "⭕" # Circle icon for Missing
            else:
                terms_incorrect += 1
                status_icon = "❌" # X icon for Incorrect

            all_field_details.append([
                status_icon,
                model_group,
                r['field'],
                format_val(r['predicted']),
                format_val(r['ground_truth'])
            ])

    print(f"Summary: {terms_correct} Correct, {terms_share_matches} Matched (as Shares), {terms_incorrect} Incorrect, {terms_missing} Missing, {terms_conflicts} Conflicts, {terms_manual_check} Manual Check")

    if all_field_details:
        # Prevent tabulate from formatting numbers automatically (which causes scientific notation)
        # Also use colalign to keep things neat
        print(tabulate(all_field_details, headers=["Match", "Group", "Field", "AI Value", "DB Value"], tablefmt="grid", disable_numparse=True))
    else:
        print("All Terms, Outcome & Dates match perfectly!")

if __name__ == "__main__":
    main()
