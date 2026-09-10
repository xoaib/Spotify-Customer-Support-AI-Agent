"""
Empirical Failure Analysis Module.
Extracts and categorizes real failures directly from evaluation_results.json.
"""

import os
import sys
import json
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import BASE_DIR

EVAL_PATH = os.path.join(BASE_DIR, "data", "processed", "evaluation_results.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "failure_analysis.json")


def analyze_failures():
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data["detailed_predictions"]
    failures = {
        "intent_misclassifications": [],
        "missed_escalations": [],
        "over_escalations": []
    }

    for p in predictions:
        # Intent mismatch
        if p["pred_intent"] != p["gold_intent"]:
            failures["intent_misclassifications"].append(p)
        # Missed escalation: gold=ESCALATE, pred=AUTO_HANDLE
        if p["gold_action"] == "ESCALATE" and p["pred_action"] == "AUTO_HANDLE":
            failures["missed_escalations"].append(p)
        # Over-escalation: gold=AUTO_HANDLE, pred=ESCALATE
        if p["gold_action"] == "AUTO_HANDLE" and p["pred_action"] == "ESCALATE":
            failures["over_escalations"].append(p)

    print(f"Total Intent Misclassifications: {len(failures['intent_misclassifications'])}")
    print(f"Missed Escalations (False Negatives): {len(failures['missed_escalations'])}")
    print(f"Over-Escalations (False Positives): {len(failures['over_escalations'])}")

    print("\n--- Top Missed Escalations (False Negatives) ---")
    for i, m in enumerate(failures["missed_escalations"], 1):
        print(f"[{i}] ID {m['id']} | Gold: {m['gold_intent']} -> Pred: {m['pred_intent']}")
        print(f"    Cust: {m['customer_message']}")
        print(f"    Why needed: {m['gold_reason']}")
        print(f"    System reason: {m['pred_reason']}\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    return failures

if __name__ == "__main__":
    analyze_failures()
