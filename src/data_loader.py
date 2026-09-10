"""
Data loader utilities for loading raw tweets, processed support cases, and golden evaluation cases.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Generator, Optional
from src.config import RAW_DATA_PATH, PROCESSED_CASES_PATH, GOLDEN_SET_PATH


def load_raw_chunks(filepath: str = RAW_DATA_PATH, chunksize: int = 250_000) -> Generator[pd.DataFrame, None, None]:
    """Yields chunks of the raw Twitter customer support dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at {filepath}. Run scripts/prepare_data.py first.")
    return pd.read_csv(filepath, chunksize=chunksize, dtype=str)


def load_processed_cases(filepath: str = PROCESSED_CASES_PATH, max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads reconstructed historical support cases from JSONL."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed cases not found at {filepath}. Run scripts/prepare_data.py first.")
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
            if max_cases and len(cases) >= max_cases:
                break
    return cases


def load_golden_set(filepath: str = GOLDEN_SET_PATH) -> List[Dict[str, Any]]:
    """Loads human-labelled golden evaluation cases."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Golden evaluation set not found at {filepath}.")
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases
