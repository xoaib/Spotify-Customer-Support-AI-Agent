"""
Data preprocessing and conversation reconstruction for Spotify customer support.
"""

import os
import re
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from src.config import RAW_DATA_PATH, PROCESSED_CASES_PATH, TARGET_BRAND
from src.taxonomy import VALID_INTENTS


def clean_tweet_text(text: str, remove_handles: bool = False) -> str:
    """
    Light text cleaner that preserves typos, abbreviations, and informal customer language.
    Optionally strips @handles for clean indexing.
    """
    if not isinstance(text, str):
        return ""
    # Normalize excessive whitespace
    cleaned = re.sub(r'[\r\n\t]+', ' ', text).strip()
    if remove_handles:
        # Strip Twitter user handles like @SpotifyCares or @115887
        cleaned = re.sub(r'@\w+', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def is_usable_case(cust_text: str, brand_text: str) -> bool:
    """
    Check if a reconstructed conversation pair is usable.
    Discards empty tweets or tweets consisting solely of @mentions.
    """
    cleaned_cust = clean_tweet_text(cust_text, remove_handles=True)
    cleaned_brand = clean_tweet_text(brand_text, remove_handles=True)
    
    if len(cleaned_cust) < 5 or len(cleaned_brand) < 5:
        return False
    return True


def heuristic_intent_tag(text: str) -> str:
    """
    High-precision heuristic tagger used ONLY for categorizing the historical retrieval pool.
    Evaluation golden labels are strictly human-labelled.
    """
    t = text.lower()
    
    # Account Access & Security
    if any(k in t for k in ["hacked", "stolen", "compromised", "password reset", "reset email", 
                            "can't log in", "cant log in", "cannot login", "login failed", 
                            "someone changed my email", "unknown device"]):
        return "account_access_security"
    
    # Subscription & Billing
    if any(k in t for k in ["charge", "charged", "billing", "refund", "subscription", "premium", 
                            "payment", "invoice", "student discount", "sheerid", "family plan", 
                            "card", "receipt", "overcharged", "double charge"]):
        return "subscription_billing"
        
    # Content Availability & Licensing
    if any(k in t for k in ["greyed out", "grayed out", "missing song", "missing album", 
                            "unavailable in my country", "licensing", "removed from spotify", 
                            "why is this album gone", "catalogue", "not available"]):
        return "content_availability"
        
    # App Crash & Bug
    if any(k in t for k in ["crash", "crashes", "crashing", "freeze", "freezes", "freezing", 
                            "black screen", "reinstall", "won't open", "wont open", 
                            "desktop app", "bug", "glitch", "error code"]):
        return "app_bug_crash"
        
    # Feature Request & Community
    if any(k in t for k in ["idea", "suggestion", "feature request", "feedback", "bring back", 
                            "block ads", "wish spotify had", "option to"]):
        return "feature_request_feedback"
        
    # Playback Issues
    if any(k in t for k in ["pause", "pausing", "skipping", "skip", "shuffle", "repeat", 
                            "stutter", "buffering", "won't play", "wont play", "offline", 
                            "stops playing", "stops after", "no sound"]):
        return "playback_issue"
        
    # Default fallback to playback_issue if general audio/app query
    return "playback_issue"


def process_dataset(
    raw_path: str = RAW_DATA_PATH,
    output_path: str = PROCESSED_CASES_PATH,
    target_brand: str = TARGET_BRAND,
    chunksize: int = 250_000,
    max_cases: Optional[int] = None
) -> int:
    """
    Reconstructs paired customer-brand conversations for the target brand
    and saves them to data/processed/support_cases.jsonl.
    """
    print(f"Loading raw dataset from {raw_path}...")
    
    # 1. Collect all brand outbound tweets and the inbound IDs they replied to
    brand_outbound = []
    inbound_ids = set()
    
    for chunk in pd.read_csv(raw_path, chunksize=chunksize, dtype=str):
        sub = chunk[chunk['author_id'] == target_brand]
        if not sub.empty:
            brand_outbound.append(sub)
            in_resp = sub['in_response_to_tweet_id'].dropna()
            inbound_ids.update(in_resp)
            
    df_brand = pd.concat(brand_outbound, ignore_index=True)
    print(f"Collected {len(df_brand):,} {target_brand} outbound tweets.")
    print(f"Searching for {len(inbound_ids):,} referenced customer tweets...")
    
    # 2. Extract customer tweets in a second pass
    customer_tweets: Dict[str, Dict[str, Any]] = {}
    parent_ids_to_fetch = set()
    
    for chunk in pd.read_csv(raw_path, chunksize=chunksize, dtype=str):
        matched = chunk[chunk['tweet_id'].isin(inbound_ids)]
        for _, row in matched.iterrows():
            customer_tweets[row['tweet_id']] = {
                'text': row['text'],
                'author_id': row['author_id'],
                'created_at': row['created_at'],
                'in_response_to_tweet_id': row['in_response_to_tweet_id']
            }
            if pd.notna(row['in_response_to_tweet_id']):
                parent_ids_to_fetch.add(row['in_response_to_tweet_id'])
                
    print(f"Found {len(customer_tweets):,} directly referenced customer tweets.")
    
    # 3. Optional third pass to fetch context for multi-turn conversations
    context_tweets: Dict[str, str] = {}
    if parent_ids_to_fetch:
        print(f"Fetching context for {len(parent_ids_to_fetch):,} parent tweets...")
        for chunk in pd.read_csv(raw_path, chunksize=chunksize, dtype=str):
            matched = chunk[chunk['tweet_id'].isin(parent_ids_to_fetch)]
            for _, row in matched.iterrows():
                context_tweets[row['tweet_id']] = row['text']
                
    # 4. Reconstruct and clean cases
    cases: List[Dict[str, Any]] = []
    seen_pairs = set()
    
    for _, row in df_brand.iterrows():
        resp_to = row['in_response_to_tweet_id']
        if resp_to not in customer_tweets:
            continue
            
        cust = customer_tweets[resp_to]
        cust_text = cust['text']
        brand_text = row['text']
        
        if not is_usable_case(cust_text, brand_text):
            continue
            
        # Deduplication key
        pair_key = (clean_tweet_text(cust_text, True), clean_tweet_text(brand_text, True))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        
        # Context if available
        context_str = ""
        parent_id = cust.get('in_response_to_tweet_id')
        if parent_id and parent_id in context_tweets:
            context_str = clean_tweet_text(context_tweets[parent_id])
            
        intent = heuristic_intent_tag(cust_text)
        
        case = {
            "case_id": f"case_{row['tweet_id']}",
            "customer_tweet_id": resp_to,
            "brand_tweet_id": row['tweet_id'],
            "customer_message": clean_tweet_text(cust_text),
            "conversation_context": context_str,
            "brand_response": clean_tweet_text(brand_text),
            "resolution": clean_tweet_text(brand_text),  # Historical brand reply serves as actual resolution
            "intent": intent
        }
        cases.append(case)
        
        if max_cases and len(cases) >= max_cases:
            break
            
    # Save to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
            
    print(f"Successfully processed {len(cases):,} cases and wrote to {output_path}.")
    return len(cases)


if __name__ == "__main__":
    process_dataset()
