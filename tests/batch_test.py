import os
import json
import argparse
import glob
from collections import defaultdict
from src.database import FinanceDB
from tests.evaluator import Evaluator
from src.models import get_fields_for_category, ExtractionResult

EXTRACTION_DEFS_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "extraction_definitions.json")

# Fields that are metadata/provenance — never part of accuracy evaluation
NEVER_EVALUATE = {"source_pages", "investors_source_pages"}


def load_source_type_categories() -> dict[str, list[str]]:
    """Returns {source_type: [category, ...]} from extraction_definitions.json."""
    with open(EXTRACTION_DEFS_PATH) as f:
        defs = json.load(f)
    mapping = defaultdict(list)
    for category, defn in defs.get("field_definitions", {}).items():
        for st in defn.get("source_types", []):
            mapping[st].append(category)
    return dict(mapping)


def _counts():
    return {"correct": 0, "incorrect": 0, "total": 0}


def _with_accuracy(c: dict) -> dict:
    c["accuracy"] = round(c["correct"] / c["total"], 4) if c["total"] else None
    return c


def _is_extracted(val) -> bool:
    return val is not None and str(val).strip() not in ("", "None")


def evaluate_single_document(
    doc_url: str,
    doc_data: dict,
    issue_id: str,
    source_type: str,
    categories_for_source_type: list[str],
    db: FinanceDB,
    evaluator: Evaluator,
    exclude_fields: list,
) -> dict:
    """Evaluate one document, only counting fields where AI produced a value."""
    db_issue = db.get_issue_data(issue_id)
    if not db_issue:
        return {"error": f"No DB data for issue {issue_id}"}

    result = {
        "issue_id": issue_id,
        "doc_url": doc_url,
        "source_type": source_type,
        "fields": {},
        "investors": None,
    }

    for category in ExtractionResult.model_fields.keys():
        if category in exclude_fields or category not in categories_for_source_type:
            continue

        if category == "investors":
            ai_investors = doc_data.get("investors", [])
            if not ai_investors:
                continue
            gt_investors = db.get_investors_data(issue_id).to_dict("records")
            inv = evaluator.compare_investors(ai_investors, gt_investors)
            result["investors"] = {
                "correct": inv["correct"],
                "incorrect": inv["incorrect"] + inv["false_positives"],
                "missing": inv["missing"],
                "total": inv["correct"] + inv["incorrect"] + inv["false_positives"],
            }
            continue

        fields = [f for f in get_fields_for_category(category) if f not in exclude_fields and f not in NEVER_EVALUATE]
        if not fields:
            continue

        ai_cat_data = doc_data.get(category, {})
        ai_eval_data = ai_cat_data if isinstance(ai_cat_data, dict) else {category: ai_cat_data}

        for r in evaluator.compare_fields(ai_eval_data, db_issue, fields):
            if not _is_extracted(r["predicted"]):
                continue  # AI didn't extract this — skip
            if r.get("needs_manual_check"):
                continue  # Can't auto-score — skip
            result["fields"][r["field"]] = {
                "category": category,
                "is_match": r.get("is_match", False),
                "is_share_match": r.get("is_share_match", False),
                "predicted": r["predicted"],
                "ground_truth": r["ground_truth"],
            }

    return result


def aggregate_results(doc_results: list[dict]) -> dict:
    """Aggregate per-document results into by_source_type and by_field summaries."""
    by_source_type: dict[str, dict] = defaultdict(_counts)
    by_field: dict[str, dict] = defaultdict(_counts)

    for doc in doc_results:
        if "error" in doc:
            continue

        st = doc.get("source_type", "Unknown")

        for field, fr in doc.get("fields", {}).items():
            by_source_type[st]["total"] += 1
            by_field[field]["total"] += 1
            if fr["is_match"]:
                by_source_type[st]["correct"] += 1
                by_field[field]["correct"] += 1
            else:
                by_source_type[st]["incorrect"] += 1
                by_field[field]["incorrect"] += 1

        inv = doc.get("investors")
        if inv and inv["total"] > 0:
            by_source_type[st]["total"] += inv["total"]
            by_source_type[st]["correct"] += inv["correct"]
            by_source_type[st]["incorrect"] += inv["incorrect"]
            by_field["investors"]["total"] += inv["total"]
            by_field["investors"]["correct"] += inv["correct"]
            by_field["investors"]["incorrect"] += inv["incorrect"]

    return {
        "by_source_type": {st: _with_accuracy(dict(c)) for st, c in by_source_type.items()},
        "by_field": {f: _with_accuracy(dict(c)) for f, c in by_field.items()},
    }


def build_issue_debug(doc_results: list[dict]) -> dict:
    """Groups results by issue, listing every incorrect field for easy debugging."""
    issues: dict[str, dict] = {}

    for doc in doc_results:
        if "error" in doc:
            issue_id = doc.get("issue_id", "unknown")
            issues.setdefault(issue_id, {"correct": 0, "incorrect": 0, "total": 0, "accuracy": None, "errors": []})
            issues[issue_id]["errors"].append({"type": "evaluation_error", "detail": doc["error"]})
            continue

        issue_id = doc["issue_id"]
        if issue_id not in issues:
            issues[issue_id] = {"correct": 0, "incorrect": 0, "total": 0, "accuracy": None, "incorrect_fields": [], "missing_investors": []}

        entry = issues[issue_id]

        for field, fr in doc.get("fields", {}).items():
            entry["total"] += 1
            if fr["is_match"]:
                entry["correct"] += 1
            else:
                entry["incorrect"] += 1
                entry["incorrect_fields"].append({
                    "field": field,
                    "category": fr["category"],
                    "source_type": doc["source_type"],
                    "predicted": fr["predicted"],
                    "ground_truth": fr["ground_truth"],
                })

        inv = doc.get("investors")
        if inv:
            entry["total"] += inv["total"]
            entry["correct"] += inv["correct"]
            entry["incorrect"] += inv["incorrect"]

    for entry in issues.values():
        total = entry.get("total", 0)
        correct = entry.get("correct", 0)
        entry["accuracy"] = round(correct / total, 4) if total else None

    return issues


def compute_totals(doc_results: list[dict]) -> dict:
    valid = [r for r in doc_results if "error" not in r]
    c = _counts()
    for doc in valid:
        for _, fr in doc.get("fields", {}).items():
            c["total"] += 1
            c["correct" if fr["is_match"] else "incorrect"] += 1
        inv = doc.get("investors")
        if inv and inv["total"] > 0:
            c["total"] += inv["total"]
            c["correct"] += inv["correct"]
            c["incorrect"] += inv["incorrect"]
    return {
        "total_documents": len(valid),
        "total_issues": len({r["issue_id"] for r in valid}),
        "errors": len(doc_results) - len(valid),
        **_with_accuracy(c),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate AI extraction accuracy by field and source type")
    parser.add_argument("-n", type=int, help="Number of issue files to process (default: all)")
    parser.add_argument("--output-dir", default="output_json", help="Directory containing AI JSON results")
    parser.add_argument("--output-file", default="batch_test_results.json", help="Output JSON file for results")
    parser.add_argument("--exclude-fields", nargs="+", default=[], help="Fields to exclude from evaluation")
    args = parser.parse_args()

    db = FinanceDB()
    evaluator = Evaluator()
    source_type_categories = load_source_type_categories()

    json_files = glob.glob(os.path.join(args.output_dir, "*_extraction.json"))
    if not json_files:
        print(f"No extraction files found in {args.output_dir}")
        return

    if args.n:
        json_files = json_files[:args.n]

    print(f"Processing {len(json_files)} files...")
    if args.exclude_fields:
        print(f"Excluding fields: {', '.join(args.exclude_fields)}")

    doc_results = []

    for json_file in json_files:
        issue_id = os.path.basename(json_file).replace("_extraction.json", "")
        print(f"Evaluating {issue_id}...")

        try:
            with open(json_file) as f:
                ai_data_full = json.load(f)
        except Exception as e:
            doc_results.append({"issue_id": issue_id, "error": str(e)})
            continue

        source_type_map = db.get_source_type_map(issue_id)

        for doc_url, doc_data in ai_data_full.items():
            doc_id = str(doc_data.get("id", ""))
            source_type = source_type_map.get(doc_url) or source_type_map.get(doc_id, "Unknown")
            categories = source_type_categories.get(source_type, [])

            if not categories:
                continue  # No categories mapped — nothing to evaluate

            result = evaluate_single_document(
                doc_url=doc_url,
                doc_data=doc_data,
                issue_id=issue_id,
                source_type=source_type,
                categories_for_source_type=categories,
                db=db,
                evaluator=evaluator,
                exclude_fields=args.exclude_fields,
            )
            doc_results.append(result)

    aggregated = aggregate_results(doc_results)
    totals = compute_totals(doc_results)

    output_data = {
        "totals": totals,
        "by_source_type": aggregated["by_source_type"],
        "by_field": aggregated["by_field"],
    }

    output_path = os.path.join(args.output_dir, args.output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    debug_filename = args.output_file.replace(".json", "_debug.json")
    debug_path = os.path.join(args.output_dir, debug_filename)
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(build_issue_debug(doc_results), f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to {output_path}")
    print(f"Debug breakdown saved to {debug_path}")
    acc_pct = f"{totals['accuracy']*100:.1f}%" if totals["accuracy"] is not None else "N/A"
    print(f"Overall accuracy: {totals['correct']}/{totals['total']} ({acc_pct}) across {totals['total_documents']} documents, {totals['total_issues']} issues\n")

    print("--- Accuracy by Source Type ---")
    for st, c in sorted(aggregated["by_source_type"].items(), key=lambda x: -(x[1]["accuracy"] or 0)):
        pct = f"{c['accuracy']*100:.1f}%" if c["accuracy"] is not None else "N/A"
        print(f"  {st:<25} {c['correct']:>4}/{c['total']:<4} ({pct})")

    print("\n--- Accuracy by Field (worst first) ---")
    field_rows = [
        (c["accuracy"] if c["accuracy"] is not None else 1.0, f, c["correct"], c["total"])
        for f, c in aggregated["by_field"].items()
        if c["total"] > 0
    ]
    for acc, field, correct, total in sorted(field_rows):
        pct = f"{acc*100:.1f}%"
        print(f"  {field:<35} {correct:>4}/{total:<4} ({pct})")


if __name__ == "__main__":
    main()
