"""
Sentence-BERT semantic matching.

Stage 2: Uses pretrained all-MiniLM-L6-v2 for semantic similarity.
This model is NOT fine-tuned — we use it as a pretrained embedding model
to capture semantic similarity between resume and job description text.
"""
import numpy as np
from typing import List, Optional


class SemanticMatcher:
    """
    Sentence-BERT semantic similarity matcher.
    
    Uses pretrained sentence-transformers/all-MiniLM-L6-v2 (~80MB).
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
    
    def _load_model(self):
        """Lazy-load the model on first use."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
    
    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Calculate semantic cosine similarity between two texts.
        """
        self._load_model()
        embeddings = self.model.encode([text_a, text_b], normalize_embeddings=True)
        sim = float(np.dot(embeddings[0], embeddings[1]))
        return float(np.clip(sim, 0.0, 1.0))
    
    def batch_similarity(self, texts_a: List[str], texts_b: List[str],
                         batch_size: int = 64) -> np.ndarray:
        """
        Calculate pairwise semantic similarity for aligned lists.
        
        Args:
            texts_a: first list of texts
            texts_b: second list of texts  
            batch_size: encoding batch size
            
        Returns:
            array of similarity scores
        """
        self._load_model()
        
        emb_a = self.model.encode(texts_a, batch_size=batch_size,
                                   normalize_embeddings=True, show_progress_bar=True)
        emb_b = self.model.encode(texts_b, batch_size=batch_size,
                                   normalize_embeddings=True, show_progress_bar=True)
        
        # Row-wise dot product (cosine similarity since embeddings are normalized)
        sims = np.sum(emb_a * emb_b, axis=1)
        return np.clip(sims, 0.0, 1.0)
    
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts into embeddings."""
        self._load_model()
        return self.model.encode(texts, batch_size=batch_size,
                                  normalize_embeddings=True, show_progress_bar=True)
