# Spotify Support Golden Evaluation Set Documentation

## 1. Overview
The Golden Evaluation Set (`golden_set.jsonl`) contains **190 human-curated and human-labelled examples** derived from real customer tweets directed to `@SpotifyCares` from the Kaggle *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`).

The evaluation set was explicitly curated by a human engineer following strict annotation guidelines to avoid synthetic LLM evaluation bias.

## 2. Sampling Strategy & Stratification
The dataset covers common, medium-frequency, rare, ambiguous, and high-risk support queries:

### Distribution by Intent
| Intent | Count | Percentage | Frequency Tier |
| :--- | :--- | :--- | :--- |
| `playback_issue` | 40 | 21.1% | High (Common) |
| `app_bug_crash` | 38 | 20.0% | High (Common) |
| `subscription_billing` | 32 | 16.8% | Medium |
| `content_availability` | 30 | 15.8% | Medium |
| `feature_request_feedback` | 26 | 13.7% | Medium / Nuanced |
| `account_access_security` | 24 | 12.6% | High-Risk / Escalation |
| **Total** | **190** | **100.0%** | |

### Distribution by Action
- **`AUTO_HANDLE`**: 134 (70.5%) — queries where safe, grounded, self-service troubleshooting or educational explanations resolve the issue.
- **`ESCALATE`**: 56 (29.5%) — queries involving account takeover, private billing/refund operations, extreme message ambiguity, legal copyright notices, or PR controversies.

## 3. Labeling Methodology & Decision Rules

1. **Grounded Brand Standard**:
   Labels reflect how official `@SpotifyCares` agents historically triage and resolve real customer problems.
2. **Conservative Escalation Policy**:
   - Every `account_access_security` issue is labelled `ESCALATE` because an automated agent cannot verify identities or safely manipulate user credentials without human review.
   - For `subscription_billing`, general informational questions (pricing, family plan member limit, student signup steps) are labelled `AUTO_HANDLE`, while actual charges, duplicate billings, and refund requests are labelled `ESCALATE`.
   - Ambiguous queries lacking issue descriptions (e.g., *"it stops"*, *"crash"*, *"broken"*) are labelled `ESCALATE` because safe automated resolution is impossible without clarifying questions.
3. **Realistic Noise Preservation**:
   All customer messages preserve realistic grammar mistakes, spelling typos (*"doesnt work"*, *"cant log in"*), informal slang, and emotional frustration to ensure true real-world robustness.

## 4. Schema
Each record in `golden_set.jsonl` adheres to:
```json
{
  "id": 1,
  "customer_message": "...",
  "context": "...",
  "gold_intent": "playback_issue",
  "gold_action": "AUTO_HANDLE",
  "gold_reason": "...",
  "reference_resolution": "..."
}
```
