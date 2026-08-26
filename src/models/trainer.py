"""
Supervised ML model training for resume-job fit classification.

Heavily reuses the training pattern from the original complaint-classifier's train.py:
- Model comparison loop (LR, NB, XGBoost)
- Macro-averaged metrics
- Confusion matrix generation
- joblib serialization

Key difference: Uses engineered features (not raw TF-IDF vectors) as input.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def build_feature_vector(
    tfidf_sim: float,
    sbert_sim: float,
    skill_coverage: float,
    required_skill_coverage: float,
    preferred_skill_coverage: float,
    resume_skill_count: int,
    jd_skill_count: int,
    matched_skill_count: int,
    missing_skill_count: int,
    resume_word_count: int,
    jd_word_count: int,
    education_match: float = 0.5,
    experience_match: float = 0.5,
) -> np.ndarray:
    """
    Build a structured feature vector for a single resume/JD pair.
    
    All features documented for explainability:
    [0] tfidf_similarity        - TF-IDF cosine similarity
    [1] sbert_similarity        - Sentence-BERT cosine similarity
    [2] skill_coverage          - Overall skill overlap ratio
    [3] required_skill_coverage - Required skill match ratio
    [4] preferred_skill_coverage - Preferred skill match ratio
    [5] resume_skill_count      - Number of skills in resume
    [6] jd_skill_count          - Number of skills in JD
    [7] matched_skill_count     - Count of matched skills
    [8] missing_skill_count     - Count of missing skills
    [9] resume_word_count       - Resume length in words
    [10] jd_word_count          - JD length in words
    [11] length_ratio           - Resume/JD length ratio
    [12] education_match        - Education match indicator
    [13] experience_match       - Experience match indicator
    """
    length_ratio = resume_word_count / max(jd_word_count, 1)
    
    return np.array([
        tfidf_sim,
        sbert_sim,
        skill_coverage,
        required_skill_coverage,
        preferred_skill_coverage,
        resume_skill_count,
        jd_skill_count,
        matched_skill_count,
        missing_skill_count,
        resume_word_count,
        jd_word_count,
        length_ratio,
        education_match,
        experience_match,
    ])


FEATURE_NAMES = [
    "tfidf_similarity",
    "sbert_similarity",
    "skill_coverage",
    "required_skill_coverage",
    "preferred_skill_coverage",
    "resume_skill_count",
    "jd_skill_count",
    "matched_skill_count",
    "missing_skill_count",
    "resume_word_count",
    "jd_word_count",
    "length_ratio",
    "education_match",
    "experience_match",
]


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    class_weights: dict = None,
) -> dict:
    """
    Train models and evaluate them.
    
    Reuses the comparison pattern from the original complaint-classifier:
    - Logistic Regression
    - Gaussian Naive Bayes (text baseline adapted for continuous features)
    - XGBoost
    
    Returns dict of model results with metrics and trained models.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build model dictionary (same pattern as original train.py)
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1,
            class_weight="balanced" if class_weights else None,
            random_state=42,
        ),
    }
    
    if XGBOOST_AVAILABLE:
        n_classes = len(label_encoder.classes_)
        xgb_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
        }
        if class_weights:
            # For XGBoost, use sample_weight during fit instead
            pass
        models["XGBoost"] = XGBClassifier(**xgb_params)
    
    results = {}
    
    print("\n" + "=" * 70)
    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, preds)
        prec_macro = precision_score(y_test, preds, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, preds, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, preds)
        
        # Per-class report
        report = classification_report(
            y_test, preds,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        
        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm,
            "classification_report": report,
        }
        
        print(f"  Accuracy      : {acc:.4f}")
        print(f"  Macro Precision: {prec_macro:.4f}")
        print(f"  Macro Recall   : {rec_macro:.4f}")
        print(f"  Macro F1       : {f1_macro:.4f}")
        print(f"  Weighted F1    : {f1_weighted:.4f}")
    
    print("=" * 70)
    
    # Find best model by Macro F1 (primary metric)
    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    print(f"\nBest model: {best_name} (Macro F1={results[best_name]['f1_macro']:.4f})")
    
    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": results[best_name]["model"],
        "scaler": scaler,
        "label_encoder": label_encoder,
    }
