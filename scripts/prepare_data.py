"""
End-to-End Data Preparation:
1. Downloads raw Customer Support on Twitter dataset (twcs.csv) if not present.
2. Filters to @SpotifyCares and reconstructs 42k+ customer-brand conversation pairs.
3. Fits and caches the TF-IDF retrieval index for sub-second retrieval.

Usage:
    python scripts/prepare_data.py
"""

import os
import sys
import time
import requests

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RAW_DATA_PATH, PROCESSED_CASES_PATH
from src.preprocessing import process_dataset
from src.retrieval import build_and_save_index

DATA_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"


def download_if_missing(output_path: str = RAW_DATA_PATH):
    """Downloads twcs.csv from CDN if not present or incomplete."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100_000_000:
        print(f"Raw dataset present at {output_path} ({os.path.getsize(output_path)/(1024*1024):.1f} MB)")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading raw dataset from {DATA_URL} to {output_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}

    with requests.get(DATA_URL, headers=headers, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024 * 4):
                if chunk:
                    f.write(chunk)
    print(f"Download finished: {output_path} ({os.path.getsize(output_path)/(1024*1024):.1f} MB)")


def main():
    print("=== Step 1: Check / Download Raw Dataset ===")
    download_if_missing(RAW_DATA_PATH)

    print("\n=== Step 2: Reconstruct Conversations for Spotify Support ===")
    t0 = time.time()
    num_cases = process_dataset(raw_path=RAW_DATA_PATH, output_path=PROCESSED_CASES_PATH)
    t1 = time.time()
    print(f"Completed conversation reconstruction: {num_cases:,} cases in {t1 - t0:.1f}s.")

    print("\n=== Step 3: Build & Cache TF-IDF Retrieval Index ===")
    build_and_save_index()
    print("\nAll data preparation complete and retrieval index ready!")


if __name__ == "__main__":
    main()
