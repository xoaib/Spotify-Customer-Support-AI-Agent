"""
Automated Evaluation Harness for Customer Support AI Agent.

Evaluates:
1. Majority Class Baseline
2. TF-IDF + Logistic Regression Baseline
3. Main Customer Support AI Agent

Computes:
- Overall Accuracy, Macro F1, Weighted F1
- Per-intent Precision, Recall, and F1
- Confusion Matrix
- Escalation Precision, Recall, Automation Rate, Escalation Rate
- Comparative summary table
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
)

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import GOLDEN_SET_PATH, BASE_DIR
from src.taxonomy import VALID_INTENTS
from evaluation.baselines import MajorityBaseline, TfidfLogisticRegressionBaseline
from src.agent import SupportAgent

EVAL_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "evaluation_results.json")


def load_golden_set(path: str = GOLDEN_SET_PATH) -> List[Dict[str, Any]]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def run_evaluation():
    print(f"Loading Golden Evaluation Set from {GOLDEN_SET_PATH}...")
    golden_cases = load_golden_set()
    n_cases = len(golden_cases)
    print(f"Loaded {n_cases} human-curated golden evaluation examples.\n")

    y_true_intent = [c["gold_intent"] for c in golden_cases]
    y_true_action = [c["gold_action"] for c in golden_cases]

    # 1. Evaluate Majority Baseline
    print("Evaluating Baseline 1: Majority Class Classifier...")
    maj_clf = MajorityBaseline()
    y_pred_maj = [maj_clf.predict(c["customer_message"])[0] for c in golden_cases]

    # 2. Evaluate TF-IDF + Logistic Regression Baseline
    print("Evaluating Baseline 2: TF-IDF + Logistic Regression...")
    lr_clf = TfidfLogisticRegressionBaseline()
    y_pred_lr = [lr_clf.predict(c["customer_message"])[0] for c in golden_cases]

    # 3. Evaluate AI Support Agent
    print("Evaluating AI Support Agent (Classification + Escalation + Retrieval)...")
    agent = SupportAgent()
    agent_results = []
    for i, c in enumerate(golden_cases, 1):
        if i % 50 == 0 or i == n_cases:
            print(f"  Processed {i}/{n_cases} cases...")
        res = agent.process_message(c["customer_message"], context=c.get("context", ""))
        agent_results.append(res)

    y_pred_agent_intent = [r["intent"] for r in agent_results]
    y_pred_agent_action = [r["action"] for r in agent_results]

    # --- Compute Metrics ---
    def calc_intent_metrics(y_true, y_pred, labels=VALID_INTENTS):
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        per_class = {}
        for idx, label in enumerate(labels):
            per_class[label] = {
                "precision": round(float(p[idx]), 4),
                "recall": round(float(r[idx]), 4),
                "f1": round(float(f1[idx]), 4),
                "support": int(s[idx])
            }
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        return {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_intent": per_class,
            "confusion_matrix": cm
        }

    maj_metrics = calc_intent_metrics(y_true_intent, y_pred_maj)
    lr_metrics = calc_intent_metrics(y_true_intent, y_pred_lr)
    agent_metrics = calc_intent_metrics(y_true_intent, y_pred_agent_intent)

    # --- Escalation Metrics for Agent ---
    # True positives: gold=ESCALATE, pred=ESCALATE
    # False positives: gold=AUTO_HANDLE, pred=ESCALATE
    # False negatives: gold=ESCALATE, pred=AUTO_HANDLE
    # True negatives: gold=AUTO_HANDLE, pred=AUTO_HANDLE
    tp = sum(1 for yt, yp in zip(y_true_action, y_pred_agent_action) if yt == "ESCALATE" and yp == "ESCALATE")
    fp = sum(1 for yt, yp in zip(y_true_action, y_pred_agent_action) if yt == "AUTO_HANDLE" and yp == "ESCALATE")
    fn = sum(1 for yt, yp in zip(y_true_action, y_pred_agent_action) if yt == "ESCALATE" and yp == "AUTO_HANDLE")
    tn = sum(1 for yt, yp in zip(y_true_action, y_pred_agent_action) if yt == "AUTO_HANDLE" and yp == "AUTO_HANDLE")

    esc_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    esc_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    esc_f1 = (2 * esc_precision * esc_recall / (esc_precision + esc_recall)) if (esc_precision + esc_recall) > 0 else 0.0

    automation_rate = (tn + fn) / n_cases
    escalation_rate = (tp + fp) / n_cases

    escalation_metrics = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "escalation_precision": round(float(esc_precision), 4),
        "escalation_recall": round(float(esc_recall), 4),
        "escalation_f1": round(float(esc_f1), 4),
        "automation_rate": round(float(automation_rate), 4),
        "escalation_rate": round(float(escalation_rate), 4)
    }

    # --- Print Evaluation Report ---
    print("\n" + "=" * 70)
    print("              INTENT CLASSIFICATION EVALUATION REPORT")
    print("=" * 70)
    print(f"{'Model':<30} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    print("-" * 70)
    print(f"{'Baseline 1 (Majority)':<30} | {maj_metrics['accuracy']*100:6.2f}%    | {maj_metrics['macro_f1']*100:6.2f}%    | {maj_metrics['weighted_f1']*100:6.2f}%")
    print(f"{'Baseline 2 (TF-IDF + LogReg)':<30} | {lr_metrics['accuracy']*100:6.2f}%    | {lr_metrics['macro_f1']*100:6.2f}%    | {lr_metrics['weighted_f1']*100:6.2f}%")
    print(f"{'AI Support Agent':<30} | {agent_metrics['accuracy']*100:6.2f}%    | {agent_metrics['macro_f1']*100:6.2f}%    | {agent_metrics['weighted_f1']*100:6.2f}%")
    print("=" * 70)

    print("\n--- Per-Intent Performance: AI Support Agent ---")
    print(f"{'Intent':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 72)
    for intent, scores in agent_metrics["per_intent"].items():
        print(f"{intent:<28} | {scores['precision']*100:6.2f}%    | {scores['recall']*100:6.2f}%    | {scores['f1']*100:6.2f}%    | {scores['support']:<8}")

    print("\n--- Confusion Matrix: AI Support Agent ---")
    header = "True \\ Pred".ljust(15) + "".join([f"{i[:7]:>8}" for i in VALID_INTENTS])
    print(header)
    for idx, row in enumerate(agent_metrics["confusion_matrix"]):
        row_str = f"{VALID_INTENTS[idx][:14]:<15}" + "".join([f"{val:>8}" for val in row])
        print(row_str)

    print("\n" + "=" * 70)
    print("               ESCALATION DECISION EVALUATION REPORT")
    print("=" * 70)
    print(f"Total Cases Evaluated:       {n_cases}")
    print(f"Golden Escalations Needed:   {tp + fn} ({((tp + fn)/n_cases)*100:.1f}%)")
    print(f"System Escalated Cases:      {tp + fp} ({escalation_rate*100:.1f}%)")
    print(f"System Auto-Handled Cases:   {tn + fn} ({automation_rate*100:.1f}%)")
    print("-" * 70)
    print(f"Escalation Precision:        {esc_precision*100:.2f}% (How many escalated cases truly needed it)")
    print(f"Escalation Recall:           {esc_recall*100:.2f}% (How many needed escalations were caught)")
    print(f"Escalation F1-Score:         {esc_f1*100:.2f}%")
    print(f"Missed Escalations (FN):     {fn} (Critical safety metric)")
    print(f"Over-Escalations (FP):       {fp} (Safe conservative trade-off)")
    print("=" * 70)

    # Save all results to disk
    full_output = {
        "n_cases": n_cases,
        "majority_baseline": maj_metrics,
        "lr_baseline": lr_metrics,
        "agent_intent": agent_metrics,
        "escalation_metrics": escalation_metrics,
        "detailed_predictions": [
            {
                "id": c["id"],
                "customer_message": c["customer_message"],
                "gold_intent": c["gold_intent"],
                "gold_action": c["gold_action"],
                "gold_reason": c["gold_reason"],
                "reference_resolution": c.get("reference_resolution", ""),
                "pred_intent": r["intent"],
                "intent_confidence": r["intent_confidence"],
                "pred_action": r["action"],
                "pred_reason": r["reason"],
                "generated_reply": r["reply"]
            }
            for c, r in zip(golden_cases, agent_results)
        ]
    }

    os.makedirs(os.path.dirname(EVAL_OUTPUT_PATH), exist_ok=True)
    with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed evaluation results saved to {EVAL_OUTPUT_PATH}")
    return full_output


if __name__ == "__main__":
    run_evaluation()
