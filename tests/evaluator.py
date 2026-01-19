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

    def compare_fields(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any], fields: List[str]) -> List[Dict[str, Any]]:
        """Compares individual fields (dates, terms, etc.) and returns all results."""
        results = []
        
        for field in fields:
            pred_val = predicted.get(field)
            gt_val = ground_truth.get(field)

            is_match = False
            if str(pred_val).strip() == str(gt_val).strip():
                is_match = True
            else:
                # Try numerical comparison if applicable
                p_num = self.normalize_amount(pred_val)
                g_num = self.normalize_amount(gt_val)
                
                if p_num is not None and g_num is not None and p_num == g_num:
                    is_match = True
            
            results.append({
                "field": field,
                "predicted": pred_val,
                "ground_truth": gt_val,
                "is_match": is_match
            })
        
        return results
