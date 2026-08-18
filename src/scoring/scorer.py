"""
Hybrid explainable scoring engine.

Combines multiple signals into a weighted match score.
All weights are application-defined heuristics loaded from config/scoring.yaml.
They are NOT industry-standard or scientifically validated.
"""
import os
import yaml
from typing import Dict, Optional


def load_scoring_config(config_path: Optional[str] = None) -> Dict:
    """Load scoring configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "scoring.yaml"
        )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def calculate_match_score(
    required_skill_coverage: float,
    semantic_similarity: float,
    tfidf_similarity: float,
    experience_relevance: float = 0.5,
    education_relevance: float = 0.5,
    preferred_skill_coverage: float = 0.0,
    ml_model_probability: float = 0.5,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Calculate the hybrid match score.
    
    All input values should be in [0, 1] range.
    
    Returns:
        dict with:
            - overall_score: 0-100 score
            - category: Strong Match / Potential Match / Weak Match
            - breakdown: dict of component scores and their weighted contributions
            - weights_used: the weights that were applied
    """
    if config is None:
        config = load_scoring_config()
    
    weights = config["weights"]
    thresholds = config["thresholds"]
    
    # Calculate weighted score
    components = {
        "required_skill_coverage": {
            "raw": round(required_skill_coverage * 100, 1),
            "weight": weights["required_skill_coverage"],
            "weighted": required_skill_coverage * weights["required_skill_coverage"],
        },
        "semantic_similarity": {
            "raw": round(semantic_similarity * 100, 1),
            "weight": weights["semantic_similarity"],
            "weighted": semantic_similarity * weights["semantic_similarity"],
        },
        "tfidf_similarity": {
            "raw": round(tfidf_similarity * 100, 1),
            "weight": weights["tfidf_similarity"],
            "weighted": tfidf_similarity * weights["tfidf_similarity"],
        },
        "experience_relevance": {
            "raw": round(experience_relevance * 100, 1),
            "weight": weights["experience_relevance"],
            "weighted": experience_relevance * weights["experience_relevance"],
        },
        "education_relevance": {
            "raw": round(education_relevance * 100, 1),
            "weight": weights["education_relevance"],
            "weighted": education_relevance * weights["education_relevance"],
        },
        "preferred_skill_coverage": {
            "raw": round(preferred_skill_coverage * 100, 1),
            "weight": weights["preferred_skill_coverage"],
            "weighted": preferred_skill_coverage * weights["preferred_skill_coverage"],
        },
        "ml_model_probability": {
            "raw": round(ml_model_probability * 100, 1),
            "weight": weights["ml_model_confidence"],
            "weighted": ml_model_probability * weights["ml_model_confidence"],
        },
    }
    
    overall = sum(c["weighted"] for c in components.values()) * 100
    overall = round(min(max(overall, 0), 100), 1)
    
    # Categorize
    if overall >= thresholds["strong_match"]:
        category = "Strong Match"
    elif overall >= thresholds["potential_match"]:
        category = "Potential Match"
    else:
        category = "Weak Match"
    
    return {
        "overall_score": overall,
        "category": category,
        "breakdown": components,
        "weights_used": weights,
        "thresholds_used": thresholds,
    }
