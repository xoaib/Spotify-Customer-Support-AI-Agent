"""
Customer Support AI Agent for Spotify (@SpotifyCares).

Pipeline:
1. Intent classification (structured JSON with confidence)
2. Historical case retrieval (TF-IDF cosine similarity)
3. Grounded reply generation (LLM strictly grounded in retrieved evidence)
4. Explainable escalation decision (conservative rules prioritizing safety)
"""

import os
import re
import json
import hashlib
import requests
from typing import Dict, Any, List, Optional, Tuple

from src.config import (
    LLM_API_KEY, LLM_API_BASE, LLM_MODEL,
    INTENT_CONFIDENCE_THRESHOLD, RETRIEVAL_SIMILARITY_THRESHOLD,
    RETRIEVAL_TOP_K, ENABLE_LLM_CACHE, CACHE_DIR
)
from src.taxonomy import VALID_INTENTS, get_taxonomy_prompt_text, get_intent_metadata
from src.retrieval import get_retriever
from src.preprocessing import clean_tweet_text
from evaluation.baselines import TfidfLogisticRegressionBaseline

CACHE_FILE = os.path.join(CACHE_DIR, "llm_cache.json")


class LLMCache:
    """Simple persistent key-value cache for LLM API responses."""
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self.cache: Dict[str, str] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)

    def set(self, key: str, value: str):
        self.cache[key] = value
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


_cache = LLMCache()


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
    """
    Calls the LLM API with persistent caching.
    Falls back to a deterministic local rule/ML generator if API key is not configured.
    """
    cache_key = hashlib.md5(f"{LLM_MODEL}:{system_prompt}:{user_prompt}".encode("utf-8")).hexdigest()
    if ENABLE_LLM_CACHE:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

    if LLM_API_KEY and LLM_API_KEY != "your_api_key_here":
        try:
            url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if ENABLE_LLM_CACHE:
                _cache.set(cache_key, content)
            return content
        except Exception as e:
            print(f"[Warning] LLM API call failed ({e}), falling back to deterministic local agent.")

    # Offline / Mock Fallback implementation
    result = fallback_local_llm(system_prompt, user_prompt)
    if ENABLE_LLM_CACHE:
        _cache.set(cache_key, result)
    return result


def fallback_local_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Deterministic offline fallback that faithfully executes the task when no external API key is provided.
    Uses classical TF-IDF classification and historical retrieval extraction.
    """
    # Check if this is an intent classification prompt
    if "classify the customer support message" in system_prompt.lower() or "output valid json with 'intent'" in system_prompt.lower():
        # Extract customer message
        match = re.search(r'Customer Message:\s*"(.*?)"', user_prompt, re.DOTALL)
        msg = match.group(1) if match else user_prompt
        
        lr_baseline = TfidfLogisticRegressionBaseline()
        pred_intent, conf = lr_baseline.predict(msg)
        
        # High-coverage domain pattern matching for Spotify customer tweets
        msg_low = msg.lower()
        if any(k in msg_low for k in [
            "hack", "stolen", "compromised", "password", "reset email", 
            "can't log in", "cant log in", "cannot login", "logged out",
            "someone changed my email", "facebook", "two-factor", "2fa",
            "phishing", "banned", "disabled", "delete my account", "unauthorized login"
        ]):
            pred_intent = "account_access_security"
            conf = 0.94
        elif any(k in msg_low for k in [
            "charged", "charge", "billing", "refund", "subscription", "premium", 
            "payment", "invoice", "student discount", "sheerid", "family plan", 
            "receipt", "overcharged", "double charge", "cost", "price", "pay"
        ]):
            pred_intent = "subscription_billing"
            conf = 0.93
        elif any(k in msg_low for k in [
            "greyed out", "grayed out", "missing song", "missing album", "missing", 
            "licensing", "removed from spotify", "album gone", "not available", 
            "unplayable", "catalogue", "why did you remove", "why is this album",
            "when is", "discography"
        ]):
            pred_intent = "content_availability"
            conf = 0.92
        elif any(k in msg_low for k in [
            "crash", "crashes", "crashing", "freeze", "freezes", "black screen", 
            "clean reinstall", "won't open", "wont open", "desktop app", "bug", 
            "glitch", "error code", "wont uninstall", "overheating"
        ]):
            pred_intent = "app_bug_crash"
            conf = 0.91
        elif any(k in msg_low for k in [
            "please add", "can you add", "would love", "bring back", "suggestion", 
            "feature request", "feedback", "shoutout", "dark mode", "block ad", 
            "sleep timer", "idea", "hate the new", "loving the"
        ]):
            pred_intent = "feature_request_feedback"
            conf = 0.90
        elif any(k in msg_low for k in [
            "pause", "pausing", "skip", "skipping", "shuffle", "repeat", 
            "stutter", "buffering", "won't play", "wont play", "stops playing", 
            "offline", "bluetooth", "airpods", "chromecast", "volume", "quiet"
        ]):
            pred_intent = "playback_issue"
            conf = 0.92

        # Penalize confidence for short/ambiguous messages
        clean_len = len(clean_tweet_text(msg, remove_handles=True))
        if clean_len < 15:
            conf = min(conf, 0.45)

        return json.dumps({"intent": pred_intent, "confidence": round(conf, 2)})

    # Otherwise it's reply generation
    if "reply" in system_prompt.lower() and ("draft" in system_prompt.lower() or "grounded" in system_prompt.lower()):
        match = re.search(r'(?:Historical )?Brand Response:\s*"(.*?)"', user_prompt, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            ref_resp = match.group(1).strip()
            # Clean author handles from historical response for new tweet
            clean_resp = re.sub(r'@\w+\s*', '', ref_resp).strip()
            if len(clean_resp) > 10:
                return json.dumps({
                    "reply": clean_resp,
                    "grounding_confidence": 0.88
                })
        return json.dumps({
            "reply": "Hey there! We'd like to help. Can you let us know your device, operating system, and Spotify version so we can suggest next steps?",
            "grounding_confidence": 0.75
        })

    # Default JSON fallback
    return json.dumps({"result": "ok"})


class SupportAgent:
    """
    Main Customer Support AI Agent pipeline.
    """
    def __init__(self):
        self.retriever = get_retriever()
        self.taxonomy_text = get_taxonomy_prompt_text()

    def classify_intent(self, customer_message: str, context: str = "") -> Tuple[str, float]:
        """
        Classifies incoming message into one of the 6 Spotify support intents.
        Returns: (intent_name, confidence)
        """
        system_prompt = (
            "Classify the incoming customer support message into exactly one of the following allowed intents for Spotify (@SpotifyCares):\n"
            f"{self.taxonomy_text}\n\n"
            "Instructions:\n"
            "1. Output valid JSON containing 'intent' and 'confidence' (float between 0.0 and 1.0).\n"
            "2. Choose strictly from the allowed intent list.\n"
            "3. If the message is short, vague, or ambiguous (e.g. 'help', 'crash', 'it stops'), assign lower confidence (< 0.60).\n"
            "Example output format:\n"
            '{"intent": "playback_issue", "confidence": 0.92}'
        )

        user_prompt = f'Customer Message: "{customer_message}"'
        if context:
            user_prompt += f'\nConversation Context: "{context}"'

        raw_output = call_llm(system_prompt, user_prompt, max_tokens=100)

        # Parse JSON output
        try:
            # Extract JSON block if wrapped in markdown
            json_match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                intent = data.get("intent", "").strip()
                confidence = float(data.get("confidence", 0.50))
                if intent in VALID_INTENTS:
                    return intent, round(confidence, 2)
        except Exception:
            pass

        # Fallback to classical baseline if LLM returned malformed JSON or unallowed intent
        lr = TfidfLogisticRegressionBaseline()
        fallback_intent, conf = lr.predict(customer_message)
        return fallback_intent, round(conf, 2)

    def retrieve_cases(self, customer_message: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        """Retrieves top-k similar historical cases."""
        return self.retriever.retrieve(customer_message, top_k=top_k)

    def generate_reply(
        self,
        customer_message: str,
        intent: str,
        retrieved_cases: List[Dict[str, Any]],
        context: str = ""
    ) -> Tuple[Optional[str], float]:
        """
        Generates grounded reply using historical support examples as evidence.
        """
        # If intent is sensitive security/account risk, do not draft automated policy replies
        if intent == "account_access_security":
            return (
                "Hi there! For account security and privacy, please send us a Direct Message with your account's email address so our team can review this securely.",
                0.90
            )

        examples_text = ""
        for i, c in enumerate(retrieved_cases[:3], 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Historical Customer Issue: \"{c['customer_message']}\"\n"
            examples_text += f"Historical Brand Response: \"{c['brand_response']}\"\n"

        system_prompt = (
            "You represent Spotify support (@SpotifyCares) on Twitter.\n"
            "Draft a concise, helpful customer support reply grounded in historical brand behavior.\n\n"
            "GUIDELINES:\n"
            "1. Use the historical examples as evidence of how Spotify resolves this issue.\n"
            "2. Do not invent policies, refund amounts, timelines, or account actions.\n"
            "3. Do not claim an action has been performed when it has not.\n"
            "4. Keep replies concise and under 280 characters for Twitter.\n"
            "5. Output valid JSON: {\"reply\": \"...\", \"grounding_confidence\": 0.85}\n"
        )

        user_prompt = (
            f"Customer Message: \"{customer_message}\"\n"
            f"Classified Intent: {intent}\n"
            f"Conversation Context: \"{context}\"\n\n"
            f"Similar Historical Spotify Support Cases:\n{examples_text}\n\n"
            "Draft the grounded reply in JSON format:"
        )

        raw_output = call_llm(system_prompt, user_prompt, max_tokens=200)

        try:
            json_match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                reply = data.get("reply", "").strip()
                grounding_conf = float(data.get("grounding_confidence", 0.75))
                return reply, round(grounding_conf, 2)
        except Exception:
            pass

        # Fallback to top retrieved historical response if available
        if retrieved_cases and retrieved_cases[0]["similarity"] > 0.30:
            top_resp = re.sub(r'@\w+', '', retrieved_cases[0]["brand_response"]).strip()
            return top_resp, round(retrieved_cases[0]["similarity"], 2)

        return None, 0.0

    def decide_escalation(
        self,
        customer_message: str,
        intent: str,
        intent_confidence: float,
        retrieved_cases: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Conservative, explainable escalation rules.
        Returns: (action, reason)
        action: 'AUTO_HANDLE' or 'ESCALATE'
        """
        msg_clean = clean_tweet_text(customer_message, remove_handles=True)
        msg_lower = msg_clean.lower()

        # Signal 1: Account Access and Security concerns (Account takeover, compromised login, stolen credentials)
        if intent == "account_access_security":
            return (
                "ESCALATE",
                "Potential account security or takeover risk requires human verification and private authentication."
            )

        # Signal 2: Legal, DMCA, or high-sensitivity PR controversy
        sensitive_keywords = ["lawsuit", "sue", "lawyer", "copyright infringement", "dmca", "illegal", "press", "journalist", "boycott", "bribe", "payola", "infringement"]
        if any(k in msg_lower for k in sensitive_keywords):
            return (
                "ESCALATE",
                "Message involves legal, copyright, or regulatory sensitivities requiring executive/legal review."
            )

        # Signal 3: Billing disputes, refund requests, or financial discrepancies
        if intent == "subscription_billing":
            faq_signals = ["how much", "how many people", "how do i apply", "can i pay with", "what happens to my saved", "annual plan", "trial work", "included with"]
            is_faq = any(f in msg_lower for f in faq_signals)
            if not is_faq:
                dispute_keywords = [
                    "charged", "charge", "refund", "unauthorized", "fraud", "overdraft", 
                    "dispute", "cancel", "cancelled", "overcharged", "rejected", "decline", 
                    "failed", "amex", "paypal", "kicked", "bank", "receipt", "deducted"
                ]
                if any(k in msg_lower for k in dispute_keywords) or "refund" in msg_lower:
                    return (
                        "ESCALATE",
                        "Disputed charges, unexpected billing, and refund requests require human verification of private financial records."
                    )

        # Signal 4: Low classification confidence
        if intent_confidence < INTENT_CONFIDENCE_THRESHOLD:
            return (
                "ESCALATE",
                f"Intent classification confidence ({intent_confidence:.2f}) is below the safe threshold ({INTENT_CONFIDENCE_THRESHOLD:.2f})."
            )

        # Signal 5: Severe hardware or data loss allegations
        if any(k in msg_lower for k in ["corrupted my", "hard drive was wiped", "wiped my", "registry", "sonos"]):
            return (
                "ESCALATE",
                "Severe hardware failure or data loss allegation requires tier-2 engineering review."
            )

        # Signal 6: Short / Ambiguous message lacking technical detail
        if len(msg_clean) < 16 and not any(k in msg_lower for k in ["song", "app", "play", "sound", "mac", "ios", "pc", "ad", "music"]):
            return (
                "ESCALATE",
                "The message is ambiguous and lacks sufficient detail to safely determine the technical issue without human clarification."
            )

        # Signal 7: Insufficient historical evidence / low retrieval similarity
        top_sim = retrieved_cases[0]["similarity"] if retrieved_cases else 0.0
        if top_sim < RETRIEVAL_SIMILARITY_THRESHOLD:
            return (
                "ESCALATE",
                f"Retrieved historical cases lack sufficient similarity (top score: {top_sim:.2f}) to ground a safe automated response."
            )

        # All safety criteria satisfied
        return (
            "AUTO_HANDLE",
            "High-confidence intent with sufficient historical support examples for safe self-service resolution."
        )

    def process_message(self, customer_message: str, context: str = "") -> Dict[str, Any]:
        """
        Executes the full agent pipeline on an incoming customer message.
        """
        # Step 1: Classify intent
        intent, intent_conf = self.classify_intent(customer_message, context)

        # Step 2: Retrieve similar historical cases
        retrieved_cases = self.retrieve_cases(customer_message, top_k=RETRIEVAL_TOP_K)

        # Step 3: Decide escalation
        action, reason = self.decide_escalation(customer_message, intent, intent_conf, retrieved_cases)

        # Step 4: Generate grounded reply
        reply = None
        if action == "AUTO_HANDLE":
            reply, _ = self.generate_reply(customer_message, intent, retrieved_cases, context)
        else:
            # If escalated, we provide an escalation referral message if appropriate
            if intent == "account_access_security":
                reply = "Hi! To protect your account security, please send us a DM with your account's email address so our team can help you securely."
            elif intent == "subscription_billing":
                reply = "We'd like to look into this billing concern for you. Please send us a private DM with your account email and details so we can assist."
            else:
                reply = None

        return {
            "intent": intent,
            "intent_confidence": intent_conf,
            "action": action,
            "reason": reason,
            "reply": reply,
            "retrieved_cases": [
                {
                    "case_id": c["case_id"],
                    "similarity": c["similarity"],
                    "historical_issue": c["customer_message"][:100],
                    "historical_response": c["brand_response"][:100]
                }
                for c in retrieved_cases
            ]
        }
