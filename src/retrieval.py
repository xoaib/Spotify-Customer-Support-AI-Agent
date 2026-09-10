"""
TF-IDF Cosine Similarity Retrieval Engine for historical Spotify support cases.
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import PROCESSED_CASES_PATH, RETRIEVAL_INDEX_PATH, RETRIEVAL_TOP_K
from src.preprocessing import clean_tweet_text


class CaseRetriever:
    """
    Retrieves historical support cases most similar to an incoming customer query
    using TF-IDF vectorization and cosine similarity.
    """
    def __init__(self, index_path: str = RETRIEVAL_INDEX_PATH, cases_path: str = PROCESSED_CASES_PATH):
        self.index_path = index_path
        self.cases_path = cases_path
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.cases: List[Dict[str, Any]] = []
        self._load_or_build_index()

    def _load_or_build_index(self):
        """Loads index from disk cache or builds if not found."""
        if os.path.exists(self.index_path):
            print(f"Loading retrieval index from cache: {self.index_path}")
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.tfidf_matrix = data["matrix"]
                self.cases = data["cases"]
            print(f"Loaded {len(self.cases):,} indexed historical cases.")
        else:
            self.build_index()

    def build_index(self):
        """Builds TF-IDF index from processed support cases."""
        print(f"Building retrieval index from {self.cases_path}...")
        if not os.path.exists(self.cases_path):
            raise FileNotFoundError(f"Processed cases file not found at {self.cases_path}. Run preprocessing first.")

        cases = []
        corpus = []
        with open(self.cases_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                case = json.loads(line)
                cases.append(case)
                # Combine customer message and context for indexable text
                text_to_index = clean_tweet_text(case["customer_message"], remove_handles=True)
                if case.get("conversation_context"):
                    text_to_index += " " + clean_tweet_text(case["conversation_context"], remove_handles=True)
                corpus.append(text_to_index)

        print(f"Fitting TF-IDF vectorizer on {len(corpus):,} support cases...")
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            stop_words="english",
            sublinear_tf=True
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)

        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.cases = cases

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "matrix": self.tfidf_matrix,
                "cases": self.cases
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Retrieval index successfully saved to {self.index_path}.")

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieves top_k historical cases most similar to query.
        Returns list of cases with similarity scores.
        """
        if not self.vectorizer or self.tfidf_matrix is None:
            raise RuntimeError("Retrieval index is not initialized.")

        clean_query = clean_tweet_text(query, remove_handles=True)
        if not clean_query.strip():
            return []

        query_vec = self.vectorizer.transform([clean_query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            case = dict(self.cases[idx])
            case["similarity"] = round(score, 4)
            results.append(case)

        return results


# Global singleton instance
_retriever_instance: Optional[CaseRetriever] = None


def get_retriever() -> CaseRetriever:
    """Returns singleton CaseRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = CaseRetriever()
    return _retriever_instance


def build_and_save_index():
    """Helper to build and save index explicitly."""
    retriever = CaseRetriever.__new__(CaseRetriever)
    retriever.index_path = RETRIEVAL_INDEX_PATH
    retriever.cases_path = PROCESSED_CASES_PATH
    retriever.build_index()
