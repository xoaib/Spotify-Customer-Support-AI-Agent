"""
Baselines for intent classification:
1. MajorityBaseline (trivial lower bound)
2. TfidfLogisticRegressionBaseline (classical NLP)
"""

import os
import json
import pickle
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import PROCESSED_CASES_PATH, BASE_DIR
from src.preprocessing import clean_tweet_text
from src.taxonomy import VALID_INTENTS

BASELINE_MODEL_PATH = os.path.join(BASE_DIR, "data", "processed", "lr_baseline.pkl")


class MajorityBaseline:
    """
    Trivial baseline: always predicts the most frequent class in the dataset.
    """
    def __init__(self, majority_intent: str = "playback_issue", confidence: float = 0.35):
        self.majority_intent = majority_intent
        self.confidence = confidence

    def predict(self, text: str) -> Tuple[str, float]:
        return self.majority_intent, self.confidence


class TfidfLogisticRegressionBaseline:
    """
    Classical NLP baseline: TF-IDF (1-2 ngrams) + Logistic Regression.
    """
    def __init__(self, model_path: str = BASELINE_MODEL_PATH):
        self.model_path = model_path
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.clf: Optional[LogisticRegression] = None
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.clf = data["clf"]
        else:
            self.train()

    def train(self, max_samples: int = 15000):
        print(f"Training TF-IDF + Logistic Regression baseline on up to {max_samples:,} cases...")
        texts = []
        labels = []

        with open(PROCESSED_CASES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                case = json.loads(line)
                intent = case.get("intent")
                if intent in VALID_INTENTS:
                    msg = clean_tweet_text(case["customer_message"], remove_handles=True)
                    if len(msg) > 5:
                        texts.append(msg)
                        labels.append(intent)
                if len(texts) >= max_samples:
                    break

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=15000,
            stop_words="english",
            sublinear_tf=True
        )
        X = vectorizer.fit_transform(texts)
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
        clf.fit(X, labels)

        self.vectorizer = vectorizer
        self.clf = clf

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "clf": self.clf}, f)
        print(f"Logistic Regression baseline saved to {self.model_path}.")

    def predict(self, text: str) -> Tuple[str, float]:
        clean_text = clean_tweet_text(text, remove_handles=True)
        if not clean_text:
            return "playback_issue", 0.30

        vec = self.vectorizer.transform([clean_text])
        probs = self.clf.predict_proba(vec)[0]
        max_idx = np.argmax(probs)
        pred_label = str(self.clf.classes_[max_idx])
        confidence = float(probs[max_idx])
        return pred_label, round(confidence, 4)
