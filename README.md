# Spotify Customer Support AI Agent

A practical, evaluation-first customer support agent built on the Kaggle *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`), focused specifically on **Spotify Support (`@SpotifyCares`)**.

Instead of overcomplicating things with multi-agent frameworks or black-box autonomous actions, this project focuses on what real customer support teams actually need:
1. **Classifying customer intent**: Grouping messages into 6 distinct, data-grounded categories.
2. **Grounding replies in history**: Reusing actual troubleshooting steps and URLs historically proven by official Spotify agents.
3. **Knowing when not to answer**: Enforcing strict, conservative escalation for account security, billing disputes, and ambiguous cries for help.
4. **Proving it works**: Benchmarking against two baseline models on a 190-case human-labelled test set.

---

## Quickstart (Under 2 Minutes)

### 1. Setup Environment
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare Data (One-Time)
Downloads the Kaggle dataset if missing, cleans 42,784 Spotify conversation pairs, and builds the TF-IDF search index:
```bash
python scripts/prepare_data.py
```

### 3. Run and Verify
```bash
# Run automated evaluation against 190 human-labelled cases (~17 seconds)
python -m evaluation.evaluate

# Test 7 real-world scenarios in the terminal (~2.5 seconds)
python scripts/test_agent.py

# Launch the interactive web UI
python app.py
```

*Note: The system runs 100% locally and offline, requiring no external API keys.*

---

## File Structure (Easy to Explain)

The codebase is organized into **3 distinct responsibilities**:

```text
customer-support-agent/
|
|-- [1] THE BRAIN (Core Agent Logic)
|   +-- src/
|       |-- agent.py              # Main SupportAgent: coordinates classifier, search, and escalation
|       |-- taxonomy.py           # Defines the 6 business intents, keyword rules, and prompts
|       |-- retrieval.py          # TF-IDF cosine similarity search across 42k+ past cases (<0.3 ms)
|       |-- preprocessing.py      # Cleans raw tweets, strips handles, reconstructs conversation pairs
|       |-- config.py             # Central thresholds (0.70 confidence floor, 0.15 retrieval similarity)
|       +-- data_loader.py        # Helpers for reading customer-brand interaction records
|
|-- [2] THE TEST BENCH (Evaluation & Ground Truth)
|   +-- evaluation/
|       |-- golden_set.jsonl      # 190 human-curated and human-labelled ground-truth customer tweets
|       |-- GOLDEN_SET.md         # Documentation on sampling strategy, tier splits, and labeling rules
|       |-- baselines.py          # Benchmark 1 (Majority Class) & Benchmark 2 (TF-IDF + LogReg)
|       |-- evaluate.py           # Main harness: computes Accuracy, Macro F1, and Escalation Recall
|       |-- judge.py              # LLM-as-a-judge rubric scoring reply quality and safety
|       +-- failure_analysis.py   # Extracts real edge cases where the model failed
|
|-- [3] THE INTERFACES (CLI & Web UI)
|   |-- app.py                    # Lightweight web server using Python's built-in http.server
|   +-- web/                      # Vanilla frontend (zero external UI framework dependencies)
|       |-- index.html            # Clean HTML5 layout with preset scenario buttons
|       |-- style.css             # Spotify-themed dark mode styling
|       +-- app.js                # Frontend logic for API calls and live DOM rendering
|   +-- scripts/
|       |-- prepare_data.py       # One-click pipeline to process data and build the search index
|       +-- test_agent.py         # Terminal script testing 7 diverse real-world customer tweets
|
|-- data/                         # Local dataset and index storage (cached locally)
|-- requirements.txt              # Minimal dependencies (pandas, numpy, scikit-learn, scipy, requests)
+-- README.md                     # Single source of truth documentation
```

### How to Explain This Architecture in an Interview:
> *"The project is divided into three simple layers:*
> * 1. **`src/` is the brain:** It handles text cleaning, intent classification, historical case retrieval, and the safety escalation gate.
> * 2. **`evaluation/` is the test bench:** It holds 190 real human-labelled tweets, two baseline models, and the automated scoring harness.
> * 3. **`app.py` and `scripts/` are the interfaces:** You can test queries directly in the terminal via `scripts/test_agent.py` or through a clean browser UI powered by `app.py`."*

---

## Problem Framing: What "Good" Means for Spotify

For `@SpotifyCares`, a good AI agent is not a chatbot that chats about everything. A good agent must:
- **Accurately distinguish technical issues**: Know whether audio stutter is an offline sync glitch, a desktop client crash, or a regional song licensing removal.
- **Provide grounded, actionable steps**: Supply verified Spotify support URLs, clean reinstall instructions, or diagnostic questions historically proven to resolve the issue.
- **Protect customer accounts and money**: Never attempt to automate account takeovers, password resets, or payment disputes over public social media.

### What We Chose Not to Build:
- **No autonomous account or financial writes**: The agent never issues refunds or modifies customer database records. Those actions require authenticated backend access and human review.
- **No heavy multi-agent frameworks**: No LangGraph, CrewAI, or AutoGen. A predictable pipeline is faster, cheaper, easier to test, and significantly safer.
- **No heavy neural vector databases**: No Chroma or Pinecone. A sparse TF-IDF index queries in 0.2ms and precisely matches exact software versions and error codes without GPU dependencies.

---

## How the Agent Works

When an incoming customer tweet arrives, it goes through 4 sequential stages:

```text
Customer Tweet
      │
      ▼
1. Intent Classification (6 data-derived intents + confidence score)
      │
      ▼
2. Historical Retrieval (TF-IDF cosine similarity across 42,784 Spotify cases)
      │
      ▼
3. Conservative Escalation Gate (Strict safety rules: security, billing, ambiguity)
      ├── If High-Risk ──> Action: ESCALATE (with explicit reason)
      └── If Safe Routine ──> Action: AUTO_HANDLE
      │
      ▼
4. Grounded Reply Generation (Synthesizes resolution using verified historical steps)
```

### The 6 Data-Derived Intents:
1. `playback_issue`: Audio pausing, skipping, offline sync, song buffering.
2. `app_bug_crash`: Desktop/mobile app freezing, crashing, clean reinstall guidance.
3. `content_availability`: Missing albums, greyed-out tracks, regional licensing.
4. `subscription_billing`: Pricing FAQs (auto-handle) vs. duplicate charges / refund disputes (escalate).
5. `account_access_security`: Stolen accounts, password resets, compromised logins (always escalate).
6. `feature_request_feedback`: UI feedback, suggestions, Spotify Idea Exchange referral.

---

## Empirical Results

Every metric below is generated by running `python -m evaluation.evaluate` against the 190 human-curated test cases in `evaluation/golden_set.jsonl`:

### Intent Classification vs. Baselines
| Model | Accuracy | Macro F1 | Weighted F1 | Latency | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1 (Majority Class)** | 21.05% | 5.80% | 7.32% | < 0.1 ms | Lower bound: always predicts `playback_issue`. |
| **Baseline 2 (TF-IDF + LogReg)** | 54.74% | 50.72% | 52.49% | 0.8 ms | Classical balanced machine learning benchmark. |
| **Our AI Support Agent** | **64.74%** | **64.15%** | **64.69%** | **22.4 ms** | +10.0% accuracy and +13.4% Macro F1 over ML baseline. |

### Escalation and Business Safety
- **Escalation Recall**: **85.71%** (successfully caught 48 of 56 critical risk cases).
- **Escalation Precision**: **64.00%** (errs on the side of caution; a safe over-escalation is far better than a leaked account or incorrect billing response).
- **Safe Automation Rate**: **60.53%** of routine queries resolved without human intervention.
- **Response Quality (Judge Rubric)**: **4.57 / 5.0** average score across Correctness (4.00), Groundedness (4.55), Helpfulness (4.73), and Safety (5.00).
- **Human vs. Judge Agreement**: 100% agreement within +/- 1 point across human-graded samples (Mean Absolute Difference = 0.24), proving the judge aligns with human standards.

---

## Top 5 Failure Modes with Real Examples and Hypotheses

1. **Attachment / Media Blindness**
   - *Example*: Customer tweets *"why is it doing this????"* with an attached screenshot of a mobile error dialog.
   - *Hypothesis*: The text-only classifier sees zero diagnostic tokens and defaults to playback or low confidence.
   - *Mitigation*: Automatically escalate any tweet under 25 characters or containing image URLs straight to human triage.

2. **Third-Party / Dual-Store Billing Conflicts**
   - *Example*: *"I'm subscribed through Apple iTunes but Spotify says I'm on Free and charged my debit card too!"*
   - *Hypothesis*: Mentions of "iTunes" and "family plan" trigger informational subscription keywords instead of dispute triggers.
   - *Mitigation*: Prioritize dispute and conflict keywords over generic subscription FAQ triggers.

3. **Permanent Username Changes**
   - *Example*: *"Hey @SpotifyCares can you change my username to @johnmusic?"*
   - *Hypothesis*: Spotify usernames are permanent database keys that cannot be modified; accounts require manual migration by a human agent. The model sounds like a benign settings query and risks offering generic profile edit steps.
   - *Mitigation*: Explicit routing rule directing all username modification requests to human escalation.

4. **Catalogue Removals Masked as Playback Errors**
   - *Example*: *"Track 4 on Kanye's album stopped playing and disappeared from my starred list!"*
   - *Hypothesis*: The phrase "stopped playing" triggers `playback_issue` rather than `content_availability`.
   - *Mitigation*: Cross-reference track and album entity removals against licensing expiration keyword patterns.

5. **Prorated Annual Contract Refund Demands**
   - *Example*: *"I cancelled my 1-year prepaid subscription after 2 months, refund the remaining 10 months!"*
   - *Hypothesis*: Matches standard cancellation FAQs in `subscription_billing`, treating it as a self-service cancellation inquiry rather than a contested contractual refund.
   - *Mitigation*: Boost escalation weights whenever words like "refund remaining", "prorated", or "months left" appear.

---

## What is Misleading About My Headline Number?

Our headline intent accuracy is **64.74%**. While solid, headline accuracy alone can be misleading:
- **Class Imbalance**: Common playback queries outnumber security concerns 4 to 1. A trivial majority model gets 21% accuracy by doing nothing useful. Macro F1 (64.15%) is a far more realistic measure because it weights rare and common intents equally.
- **Accuracy Does Not Measure Safety**: Classifying a duplicate charge as `subscription_billing` counts as correct for accuracy, but if the agent auto-replies instead of escalating to billing, it fails the customer. Escalation recall (85.71%) is far more important for business safety than raw classification accuracy.

---

## What We'd Do Next with One More Week

1. **Multimodal Screenshot Ingestion**: Add a lightweight vision model to parse mobile screenshots and error codes attached to customer tweets.
2. **Multi-Turn Thread Context**: Expand conversation reconstruction to parse 3+ turn customer dialogues rather than single-turn prompt-response pairs.
3. **Dynamic Few-Shot In-Context Grounding**: Pass the top-3 retrieved historical pairs as few-shot in-context examples to draft hyper-personalized replies.
4. **Automated Error Spike Detection**: Implement automated drift detection to catch sudden spikes in novel error codes (e.g., following a broken Spotify release) before automation precision drops.
5. **Agent Copilot Assist Mode**: Build a human-in-the-loop review queue where human reps can review, edit, or approve the agent's drafted replies with one keystroke.

---

## Decision Log (12 Non-Obvious Engineering Decisions)

1. **Selected Spotify (`@SpotifyCares`) over Delta, Apple, or Amazon**: Spotify resolves technical issues publicly on Twitter with verifiable URLs and diagnostic steps, whereas airlines immediately hide behind DMs for ticket numbers and Apple redirects to static web forms.
2. **TF-IDF Search instead of Dense Neural Embeddings**: Neural embeddings require GPU memory, slow down cold start, and require multi-gigabyte models. TF-IDF indexes in 12s, queries in <0.3ms, and matches exact software version and error tokens cleanly.
3. **Conservative Escalation Bias (Safety Over Automation Rate)**: An unnecessary human escalation costs 30 seconds of triage; an automated answer to a hacked account or billing dispute causes customer churn and legal risk.
4. **Mandatory Credential Escalation**: All `account_access_security` queries are 100% escalated. Automated systems should never handle credentials or authentication over social media.
5. **190 Real Human-Curated Examples over Synthetic Data**: Curated real customer tweets with raw typos, slang, and emotional frustration to ensure true real-world robustness.
6. **Preserving Customer Typos and Informal Slang**: Did not run aggressive autocorrect on customer input because real support queries say *"cant log in"*, *"app keeps crashin"*, and *"ur app broke"*; preprocessing preserves this distribution.
7. **0.70 Intent Confidence Threshold**: Set a conservative confidence floor below which queries automatically escalate to prevent false or ungrounded responses.
8. **0.15 Retrieval Similarity Threshold**: If the closest historical match has similarity < 0.15, the problem is novel or out-of-distribution, triggering escalation.
9. **Decoupled Escalation Action from Intent**: Disconnected intent classification from the escalation decision. `subscription_billing` can be `AUTO_HANDLE` (pricing FAQ) or `ESCALATE` (duplicate charge).
10. **Self-Contained Local Execution**: Built deterministic rule-based grounding and local caching so anyone can evaluate the repository in seconds without external token costs or network latency.
11. **Filtered Brand Agent Signatures**: Removed agent handles and sign-offs (`@SpotifyCares`, `/AG`, `/RS`) from the training vocabulary so the classifier doesn't memorize agent signatures as customer features.
12. **Macro F1 as Primary Quality Metric**: Refused to optimize for raw accuracy due to the 4:1 imbalance between playback bugs and security issues.
