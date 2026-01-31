import os
import json
import argparse
import glob
from src.database import FinanceDB
from tests.evaluator import Evaluator
from src.models import get_fields_for_category, ExtractionResult

def evaluate_single_issue(issue_id: str, output_dir: str, db: FinanceDB, evaluator: Evaluator):
    """Evaluate a single issue and return accuracy metrics."""
    ai_file = os.path.join(output_dir, f"{issue_id}_extraction.json")
    
    if not os.path.exists(ai_file):
        return None
    
    try:
        # Load AI Data
        with open(ai_file, 'r') as f:
            ai_data_full = json.load(f)
        
        # Merge AI data across documents
        merged_ai_data = {}
        for doc_key, data in ai_data_full.items():
            for field, value in data.items():
                if isinstance(value, dict):
                    merged_ai_data.setdefault(field, {}).update(value)
                elif isinstance(value, list):
                    merged_ai_data.setdefault(field, []).extend(value)
                else:
                    merged_ai_data[field] = value
        
        # Load DB Ground Truth
        db_issue = db.get_issue_data(issue_id)
        db_investors = db.get_investors_data(issue_id)
        
        if not db_issue:
            return None
        
        # Evaluate Investors
        ai_investors = merged_ai_data.get("investors", [])
        gt_investors = db_investors.to_dict('records')
        inv_report = evaluator.compare_investors(ai_investors, gt_investors)
        
        # Evaluate Terms, Outcome & Dates
        fields_to_check = {
            field_name: get_fields_for_category(field_name)
            for field_name in ExtractionResult.model_fields.keys()
            if field_name not in ["investors"]
        }
        
        terms_correct = 0
        terms_share_matches = 0
        terms_incorrect = 0
        terms_missing = 0
        terms_manual_check = 0
        
        for model_group, fields in fields_to_check.items():
            ai_group_data = merged_ai_data.get(model_group, {})
            if not isinstance(ai_group_data, dict):
                ai_eval_data = {model_group: ai_group_data}
            else:
                ai_eval_data = ai_group_data
            
            field_results = evaluator.compare_fields(ai_eval_data, db_issue, fields)
            
            for r in field_results:
                if r.get('needs_manual_check'):
                    terms_manual_check += 1
                elif r['is_match']:
                    if r.get('is_share_match'):
                        terms_share_matches += 1
                    else:
                        terms_correct += 1
                else:
                    is_missing = (r['predicted'] is None or str(r['predicted']).strip() == "") and \
                                 (r['ground_truth'] is not None and str(r['ground_truth']).strip() != "")
                    if is_missing:
                        terms_missing += 1
                    else:
                        terms_incorrect += 1
        
        return {
            "issue_id": issue_id,
            "investors": {
                "correct": inv_report['correct'],
                "incorrect": inv_report['incorrect'],
                "missing": inv_report['missing'],
                "false_positives": inv_report['false_positives']
            },
            "terms_dates_outcome": {
                "correct": terms_correct,
                "share_matches": terms_share_matches,
                "incorrect": terms_incorrect,
                "missing": terms_missing,
                "manual_check": terms_manual_check
            }
        }
    except Exception as e:
        return {
            "issue_id": issue_id,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate AI extractions against SQL Database")
    parser.add_argument("-n", type=int, help="Number of files to process (default: all)")
    parser.add_argument("--output-dir", default="output_json", help="Directory containing AI JSON results")
    parser.add_argument("--output-file", default="batch_test_results.json", help="Output JSON file for results")
    args = parser.parse_args()
    
    db = FinanceDB()
    evaluator = Evaluator()
    
    # Find all extraction JSON files
    pattern = os.path.join(args.output_dir, "*_extraction.json")
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f"No extraction files found in {args.output_dir}")
        return
    
    # Limit number if specified
    if args.n:
        json_files = json_files[:args.n]
    
    print(f"Processing {len(json_files)} files...")
    
    results = []
    for json_file in json_files:
        # Extract issue_id from filename
        filename = os.path.basename(json_file)
        issue_id = filename.replace("_extraction.json", "")
        
        print(f"Evaluating {issue_id}...")
        result = evaluate_single_issue(issue_id, args.output_dir, db, evaluator)
        
        if result:
            results.append(result)
    
    # Calculate totals
    totals = {
        "total_issues": len(results),
        "investors": {
            "correct": sum(r.get("investors", {}).get("correct", 0) for r in results if "error" not in r),
            "incorrect": sum(r.get("investors", {}).get("incorrect", 0) for r in results if "error" not in r),
            "missing": sum(r.get("investors", {}).get("missing", 0) for r in results if "error" not in r),
            "false_positives": sum(r.get("investors", {}).get("false_positives", 0) for r in results if "error" not in r)
        },
        "terms_dates_outcome": {
            "correct": sum(r.get("terms_dates_outcome", {}).get("correct", 0) for r in results if "error" not in r),
            "share_matches": sum(r.get("terms_dates_outcome", {}).get("share_matches", 0) for r in results if "error" not in r),
            "incorrect": sum(r.get("terms_dates_outcome", {}).get("incorrect", 0) for r in results if "error" not in r),
            "missing": sum(r.get("terms_dates_outcome", {}).get("missing", 0) for r in results if "error" not in r),
            "manual_check": sum(r.get("terms_dates_outcome", {}).get("manual_check", 0) for r in results if "error" not in r)
        }
    }
    
    # Save results
    output_data = {
        "totals": totals,
        "results": results
    }
    
    output_path = os.path.join(args.output_dir, args.output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_path}")
    print(f"Total issues processed: {totals['total_issues']}")

if __name__ == "__main__":
    main()
