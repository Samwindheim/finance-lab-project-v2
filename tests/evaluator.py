import json
import math
import re
from decimal import Decimal, InvalidOperation
from typing import List, Dict, Any, Optional
from rapidfuzz import fuzz

class Evaluator:
    def __init__(self, similarity_threshold: int = 70):
        self.similarity_threshold = similarity_threshold

    def normalize_name(self, name: str) -> str:
        """Normalizes names for reliable comparison."""
        if not name:
            return ""
        name = name.lower()
        name = name.replace('é', 'e')
        # Remove common legal suffixes and non-alphanumeric chars
        name = re.sub(r'\b(ab|aktiebolag|ltd|inc|llc)\b', '', name)
        name = re.sub(r'[^a-z0-9]+', '', name)
        return name.strip()

    def normalize_amount(self, amount: Any) -> Optional[Decimal]:
        """Normalizes amount to a Decimal."""
        if amount is None or amount == "" or (isinstance(amount, float) and math.isnan(amount)):
            return None
        try:
            # Handle strings with spaces or commas
            clean_val = str(amount).replace(" ", "").replace(",", "")
            # If the string itself is 'nan' or 'None'
            if clean_val.lower() in ['nan', 'none']:
                return None
            return Decimal(clean_val)
        except (InvalidOperation, TypeError, ValueError):
            return None

    def compare_investors(self, predicted: List[Dict], ground_truth: List[Dict]) -> Dict[str, Any]:
        """Compares two lists of investors and returns a report."""
        report = {
            "correct": 0,
            "incorrect": 0,
            "missing": 0,
            "false_positives": 0,
            "details": []
        }

        gt_unmatched = list(ground_truth)
        
        for pred in predicted:
            best_match = self._find_best_match(pred, gt_unmatched)

            if not best_match:
                report["false_positives"] += 1
                report["details"].append({
                    "type": "False Positive",
                    "predicted": pred,
                    "ground_truth": None
                })
            else:
                gt_unmatched.remove(best_match)
                errors = self._get_investor_errors(pred, best_match)
                
                if not errors:
                    report["correct"] += 1
                    report["details"].append({
                        "type": "Correct",
                        "predicted": pred,
                        "ground_truth": best_match
                    })
                else:
                    report["incorrect"] += 1
                    report["details"].append({
                        "type": "Incorrect Value",
                        "predicted": pred,
                        "ground_truth": best_match,
                        "errors": errors
                    })

        report["missing"] = len(gt_unmatched)
        for missing in gt_unmatched:
            report["details"].append({
                "type": "Missing",
                "predicted": None,
                "ground_truth": missing
            })

        return report

    def _find_best_match(self, pred: Dict, potential_matches: List[Dict]) -> Optional[Dict]:
        best_match = None
        min_errors = float('inf')
        max_similarity = -1

        pred_norm = self.normalize_name(pred.get("name", ""))

        for gt in potential_matches:
            gt_norm = self.normalize_name(gt.get("name", ""))
            similarity = fuzz.token_sort_ratio(pred_norm, gt_norm)

            # Check for name containment (e.g., "Sven Olof Kulldorf" in "Sven Olof Kulldorf / Bank")
            # We check if one normalized name is a substring of the other
            is_contained = (gt_norm in pred_norm and len(gt_norm) > 5) or \
                          (pred_norm in gt_norm and len(pred_norm) > 5)

            if similarity < self.similarity_threshold and not is_contained:
                continue

            errors = self._get_investor_errors(pred, gt)
            num_errors = len(errors)

            if num_errors < min_errors:
                min_errors = num_errors
                max_similarity = similarity
                best_match = gt
            elif num_errors == min_errors and similarity > max_similarity:
                max_similarity = similarity
                best_match = gt

        return best_match

    def _get_investor_errors(self, pred: Dict, gt: Dict) -> List[str]:
        errors = []
        
        # Level check
        if pred.get("level") != gt.get("level") and gt.get("level") is not None:
            errors.append(f"Level mismatch: Got {pred.get('level')}, expected {gt.get('level')}")

        # Cash amount check (with integer rounding tolerance)
        p_cash = self.normalize_amount(pred.get("amount_in_cash"))
        g_cash = self.normalize_amount(gt.get("amount_in_cash"))

        if p_cash is not None and g_cash is not None:
            if int(p_cash) != int(g_cash):
                errors.append(f"Cash mismatch: Got {p_cash}, expected {g_cash}")

        # Percentage check (with small decimal tolerance)
        p_pct = self.normalize_amount(pred.get("amount_in_percentage"))
        g_pct = self.normalize_amount(gt.get("amount_in_percentage"))

        if p_pct is not None and g_pct is not None:
            if abs(float(p_pct) - float(g_pct)) > 0.01:
                errors.append(f"Percentage mismatch: Got {p_pct}%, expected {g_pct}%")

        return errors

    def detect_conflicts(self, ai_data_full: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Identifies fields that have different values across multiple documents.
        Returns a dict mapping category -> field -> list of {doc, val}
        """
        conflicts = {}
        # category -> field -> {val -> [docs]}
        registry = {}

        for doc_name, doc_data in ai_data_full.items():
            for category, category_data in doc_data.items():
                if category in ["issue_id", "id", "source_document", "source_pages"]:
                    continue
                
                if isinstance(category_data, dict):
                    for field, value in category_data.items():
                        if value is None or value == "":
                            continue
                        
                        reg_key = f"{category}.{field}"
                        registry.setdefault(reg_key, {})
                        
                        # Use string representation for registry key to handle unhashable types
                        val_str = str(value).strip()
                        registry[reg_key].setdefault(val_str, []).append({"doc": doc_name, "val": value})
                
                elif isinstance(category_data, list):
                    # For lists like investors, we look for same name AND level but different amounts
                    if category == "investors":
                        for inv in category_data:
                            name = self.normalize_name(inv.get("name", ""))
                            level = inv.get("level")
                            if not name: continue
                            
                            # Inclusion of level in key prevents valid multi-level entries from being conflicts
                            reg_key = f"investors.{name}.{level}"
                            registry.setdefault(reg_key, {})
                            
                            # We'll just serialize the whole investor dict for comparison (minus source info)
                            inv_clean = {k: v for k, v in inv.items() if k not in ["source_document", "source_pages"]}
                            val_str = json.dumps(inv_clean, sort_keys=True)
                            registry[reg_key].setdefault(val_str, []).append({"doc": doc_name, "val": inv})

        # Now identify where we have different values across DIFFERENT documents
        for key, values in registry.items():
            if len(values) > 1:
                # We have multiple unique values. 
                # Check if these values come from at least two different documents.
                all_docs = set()
                for doc_list in values.values():
                    for d in doc_list:
                        all_docs.add(d['doc'])
                
                if len(all_docs) < 2:
                    continue # Skip if all differences are within a single document
                
                parts = key.split(".")
                category = parts[0]
                # Reconstruct field name (it might contain dots if it's an investor name)
                field = ".".join(parts[1:])
                
                # Flatten the values for the report
                flat_values = []
                for doc_list in values.values():
                    flat_values.extend(doc_list)
                
                conflicts.setdefault(category, {})[field] = flat_values

        return conflicts

    def compare_fields(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any], fields: List[str]) -> List[Dict[str, Any]]:
        """Compares individual fields (dates, terms, etc.) and returns all results."""
        results = []
        
        # Get shares_in_unit for unit/share conversion check
        shares_in_unit = self.normalize_amount(ground_truth.get("shares_in_unit"))
        if shares_in_unit is None or shares_in_unit <= 0:
            shares_in_unit = Decimal("1")

        # Fields that might be reported in shares instead of units
        unit_fields = {
            "offered_units", "offered_units_unlisted", "secondary_offering", 
            "over_allotment_size", "unit_sub_total", "unit_sub_with_rights", 
            "unit_sub_without_rights", "unit_sub_guarantor", "secondary_sub",
            "over_allotment_allocation", "over_allotment_allocation_secondary"
        }
        
        for field in fields:
            pred_val = predicted.get(field)
            gt_val = ground_truth.get(field)

            is_match = False
            is_share_match = False
            
            if str(pred_val).strip() == str(gt_val).strip():
                is_match = True
            else:
                # Try numerical comparison if applicable
                p_num = self.normalize_amount(pred_val)
                g_num = self.normalize_amount(gt_val)
                
                if p_num is not None and g_num is not None:
                    if p_num == g_num:
                        is_match = True
                    elif field in unit_fields and shares_in_unit > 1:
                        # Check if predicted is actually the share count (DB value * shares_in_unit)
                        # This happens when the AI extracts the 'aktier' count instead of 'units', and 1 share != 1 unit
                        if p_num == (g_num * shares_in_unit):
                            is_match = True
                            is_share_match = True
            
            results.append({
                "field": field,
                "predicted": pred_val,
                "ground_truth": gt_val,
                "is_match": is_match,
                "is_share_match": is_share_match
            })
        
        return results
