import os
import json
import argparse
from decimal import Decimal
from src.database import FinanceDB
from tests.evaluator import Evaluator
from tabulate import tabulate
from src.models import get_fields_for_category, ExtractionResult
from tests.batch_test import load_source_type_categories, NEVER_EVALUATE

def log_header(text):
    print(f"\n{'='*20} {text} {'='*20}")

def format_val(val):
    if val is None or str(val).strip() == "":
        return "-"
    try:
        s_val = str(val).replace(" ", "").replace(",", "")
        num = Decimal(s_val)
        if num == num.to_integral_value():
            return f"{num:,.0f}".replace(",", " ")
        return f"{num:f}".rstrip("0").rstrip(".")
    except Exception:
        return str(val)


def _is_extracted(val) -> bool:
    return val is not None and str(val).strip() not in ("", "None")


def evaluate_document(doc_url, doc_data, source_type, categories, db_issue, db, evaluator):
    """Evaluate a single document. Only counts/displays fields where AI extracted a value."""
    field_details = []
    investor_report = None
    counts = {"correct": 0, "incorrect": 0}

    for category in ExtractionResult.model_fields.keys():
        if category not in categories:
            continue

        if category == "investors":
            ai_investors = doc_data.get("investors", [])
            if ai_investors:
                gt_investors = db.get_investors_data(db_issue["id"]).to_dict("records")
                investor_report = evaluator.compare_investors(ai_investors, gt_investors)
            continue

        fields = [f for f in get_fields_for_category(category) if f not in NEVER_EVALUATE]
        if not fields:
            continue

        ai_cat_data = doc_data.get(category, {})
        ai_eval_data = ai_cat_data if isinstance(ai_cat_data, dict) else {category: ai_cat_data}

        for r in evaluator.compare_fields(ai_eval_data, db_issue, fields):
            if not _is_extracted(r["predicted"]):
                continue  # AI didn't extract — skip

            if r.get("needs_manual_check"):
                counts.setdefault("manual_check", 0)
                counts["manual_check"] += 1
                field_details.append([
                    "🔍", category, r["field"],
                    format_val(r["predicted"]),
                    f"DB has {r['counterpart_field']}={format_val(r['counterpart_value'])}",
                ])
                continue

            if r["is_match"]:
                counts["correct"] += 1
            else:
                counts["incorrect"] += 1
                field_details.append([
                    "❌", category, r["field"],
                    format_val(r["predicted"]),
                    format_val(r["ground_truth"]),
                ])

    return counts, field_details, investor_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI extraction against SQL Database Ground Truth")
    parser.add_argument("--issue-id", required=True, help="The issue_id to evaluate")
    parser.add_argument("--output-dir", default="output_json", help="Directory containing AI JSON results")
    args = parser.parse_args()

    db = FinanceDB()
    evaluator = Evaluator()
    source_type_categories = load_source_type_categories()

    issue_id = args.issue_id
    ai_file = os.path.join(args.output_dir, f"{issue_id}_extraction.json")

    if not os.path.exists(ai_file):
        print(f"Error: AI extraction file not found: {ai_file}")
        return

    with open(ai_file) as f:
        ai_data_full = json.load(f)

    db_issue = db.get_issue_data(issue_id)
    if not db_issue:
        print(f"Error: No data found in 'issues' table for ID: {issue_id}")
        return

    source_type_map = db.get_source_type_map(issue_id)

    conflicts = evaluator.detect_conflicts(ai_data_full)
    conflict_table = []
    for category, fields in conflicts.items():
        for field, instances in fields.items():
            if field in NEVER_EVALUATE:
                continue
            for inst in instances:
                doc = inst["doc"]
                conflict_table.append([
                    category,
                    field,
                    doc[:40] + "..." if len(doc) > 40 else doc,
                    inst["val"],
                ])
    if conflict_table:
        log_header("MULTI-DOCUMENT CONFLICTS DETECTED")
        print(tabulate(conflict_table, headers=["Category", "Field", "Source Document", "Conflicting Value"], tablefmt="grid"))

    issue_counts = {"correct": 0, "incorrect": 0}
    issue_investor_totals = {"correct": 0, "incorrect": 0, "false_positives": 0, "missing": 0}

    for doc_url, doc_data in ai_data_full.items():
        doc_id = str(doc_data.get("id", ""))
        source_type = source_type_map.get(doc_url) or source_type_map.get(doc_id, "Unknown")
        categories = source_type_categories.get(source_type, [])

        short_url = doc_url[-60:] if len(doc_url) > 60 else doc_url
        log_header(f"DOC: ...{short_url} [{source_type}]")

        if not categories:
            print(f"  No categories mapped for source_type={source_type!r} — skipping evaluation.")
            continue

        print(f"  Evaluating categories: {', '.join(categories)}")

        counts, field_details, investor_report = evaluate_document(
            doc_url, doc_data, source_type, categories, db_issue, db, evaluator
        )

        if investor_report:
            extracted = investor_report["correct"] + investor_report["incorrect"] + investor_report["false_positives"]
            print(f"\n  Investors: {investor_report['correct']} Correct, "
                  f"{investor_report['incorrect']} Incorrect, "
                  f"{investor_report['false_positives']} False Positives, "
                  f"{investor_report['missing']} Missing "
                  f"(of {extracted} extracted)")
            for k in ("correct", "incorrect", "false_positives", "missing"):
                issue_investor_totals[k] += investor_report.get(k, 0)

            non_correct = [d for d in investor_report["details"] if d["type"] != "Correct"]
            if non_correct:
                inv_table = []
                for d in sorted(non_correct, key=lambda x: x["type"]):
                    if d["type"] == "Missing":
                        icon = "⭕"
                    elif d["type"] == "Incorrect Value":
                        icon = "❌"
                    else:
                        icon = "➕"
                    inv_table.append([
                        f"{icon} {d['type']}",
                        d["predicted"].get("name") if d["predicted"] else "N/A",
                        d["ground_truth"].get("name") if d["ground_truth"] else "N/A",
                        ", ".join(d.get("errors", [])),
                    ])
                print(tabulate(inv_table, headers=["Status", "AI Found", "DB Ground Truth", "Errors"], tablefmt="grid"))

        manual = counts.get("manual_check", 0)
        total_extracted = counts["correct"] + counts["incorrect"]
        acc = f"{100*counts['correct']/total_extracted:.1f}%" if total_extracted else "N/A"
        manual_str = f", {manual} Manual check" if manual else ""
        print(f"\n  Fields: {counts['correct']} Correct, {counts['incorrect']} Incorrect{manual_str} "
              f"(of {total_extracted} scored) — {acc}")

        if field_details:
            print(tabulate(field_details, headers=["", "Category", "Field", "AI Value", "DB Value"], tablefmt="grid", disable_numparse=True))

        for k in issue_counts:
            issue_counts[k] += counts.get(k, 0)

    log_header(f"ISSUE SUMMARY: {issue_id}")
    total_extracted = issue_counts["correct"] + issue_counts["incorrect"]
    acc = f"{100*issue_counts['correct']/total_extracted:.1f}%" if total_extracted else "N/A"
    print(f"Fields    — {issue_counts['correct']} Correct, {issue_counts['incorrect']} Incorrect "
          f"/ {total_extracted} extracted ({acc})")
    if any(issue_investor_totals.values()):
        inv_extracted = issue_investor_totals["correct"] + issue_investor_totals["incorrect"] + issue_investor_totals["false_positives"]
        inv_acc = f"{100*issue_investor_totals['correct']/inv_extracted:.1f}%" if inv_extracted else "N/A"
        print(f"Investors — {issue_investor_totals['correct']} Correct, "
              f"{issue_investor_totals['incorrect']} Incorrect, "
              f"{issue_investor_totals['false_positives']} False Positives, "
              f"{issue_investor_totals['missing']} Missing "
              f"/ {inv_extracted} extracted ({inv_acc})")


if __name__ == "__main__":
    main()
