"""
Configuration settings for Customer Support AI Agent.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "twcs.csv")
PROCESSED_CASES_PATH = os.path.join(BASE_DIR, "data", "processed", "support_cases.jsonl")
RETRIEVAL_INDEX_PATH = os.path.join(BASE_DIR, "data", "processed", "retrieval_index.pkl")
GOLDEN_SET_PATH = os.path.join(BASE_DIR, "evaluation", "golden_set.jsonl")
CACHE_DIR = os.getenv("CACHE_DIR", os.path.join(BASE_DIR, "data", "cache"))

# Target Brand
TARGET_BRAND = "SpotifyCares"

# Model Configuration
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "")
LLM_MODEL = os.getenv("LLM_MODEL", "local")

# System & Escalation Thresholds
# If confidence falls below this threshold, escalate
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.70"))
# Minimum retrieval cosine similarity threshold to consider grounding evidence valid
RETRIEVAL_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.15"))
# Top-K historical cases to retrieve for grounding
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# Caching Configuration
ENABLE_LLM_CACHE = os.getenv("ENABLE_LLM_CACHE", "true").lower() in ("1", "true", "yes")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PROCESSED_CASES_PATH), exist_ok=True)
