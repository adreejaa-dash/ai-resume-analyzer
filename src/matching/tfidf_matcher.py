"""
TF-IDF based text matching.

Stage 1 baseline: Classical NLP approach using TF-IDF vectorization
and cosine similarity. Reuses the vectorizer pattern from the original
complaint-classifier's train.py.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import joblib


class TfidfMatcher:
    """
    TF-IDF cosine similarity matcher.
    
    Uses the same vectorizer configuration as the original complaint-classifier:
    ngram_range=(1,2), sublinear_tf=True
    """
    
    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
        )
        self.is_fitted = False
    
    def fit(self, texts: list):
        """
        Fit the TF-IDF vectorizer on training texts.
        
        IMPORTANT: Must be fitted ONLY on training data to prevent data leakage.
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
    
    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        If the vectorizer hasn't been fitted on a corpus,
        fits on just these two texts (for inference/demo mode).
        """
        if not self.is_fitted:
            # Inference mode: fit on the two input texts
            vectors = TfidfVectorizer(
                ngram_range=(1, 2), max_features=10000, sublinear_tf=True
            ).fit_transform([text_a, text_b])
        else:
            vectors = self.vectorizer.transform([text_a, text_b])
        
        sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    
    def batch_similarity(self, texts_a: list, texts_b: list) -> np.ndarray:
        """
        Calculate pairwise cosine similarity for aligned lists.
        Returns array of similarity scores.
        
        Requires the vectorizer to be fitted first.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before batch similarity.")
        
        vecs_a = self.vectorizer.transform(texts_a)
        vecs_b = self.vectorizer.transform(texts_b)
        
        # Row-wise cosine similarity
        sims = np.array([
            cosine_similarity(vecs_a[i:i+1], vecs_b[i:i+1])[0][0]
            for i in range(len(texts_a))
        ])
        return np.clip(sims, 0.0, 1.0)
    
    def save(self, path: str):
        """Save the fitted vectorizer."""
        joblib.dump(self.vectorizer, path)
    
    def load(self, path: str):
        """Load a fitted vectorizer."""
        self.vectorizer = joblib.load(path)
        self.is_fitted = True
