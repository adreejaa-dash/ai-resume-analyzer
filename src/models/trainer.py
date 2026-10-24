"""
Supervised ML model training for resume–job fit classification.

Trains and evaluates Logistic Regression and XGBoost classifiers on
structured NLP feature vectors derived from resume–JD pairs.

Pipeline:
  - Build 14-dimensional feature vectors (TF-IDF sim, SBERT sim, skill
    coverage, experience/education indicators, word-count ratios)
  - Log-normalisation applied to word-count features (skew reduction)
  - StandardScaler normalisation (fit on train only)
  - Model comparison: Logistic Regression vs XGBoost (if available)
  - Primary metric: Macro F1 (handles severe class imbalance)
  - joblib serialisation of best model, scaler, and label encoder

Phase 2 changes:
  - XGBoost now receives sample_weight (fixes silent class-imbalance bug)
  - Log-normalisation applied to resume_word_count and jd_word_count
  - education_match and experience_match removed from training feature
    vector to eliminate the training/inference mismatch identified in audit
    (they defaulted to 0.5 for every training row but were real values at
    inference; keeping them constant adds noise, not signal)
  - Feature vector is now 12-dimensional (was 14)
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
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
    # NOTE: education_match and experience_match removed from the feature
    # vector in Phase 2.  They were always 0.5 during training but real
    # values at inference, creating a training/inference mismatch.  Keeping
    # them as keyword-only **kwargs preserves backward-compatible call sites
    # in the Streamlit app without breaking anything.
    **kwargs,
) -> np.ndarray:
    """
    Build a 12-dimensional structured feature vector for a single resume/JD pair.

    All features documented for explainability:
    [0]  tfidf_similarity        - TF-IDF cosine similarity
    [1]  sbert_similarity        - Sentence-BERT cosine similarity
    [2]  skill_coverage          - Overall skill overlap ratio
    [3]  required_skill_coverage - Required skill match ratio
    [4]  preferred_skill_coverage - Preferred skill match ratio
    [5]  resume_skill_count      - Number of skills in resume
    [6]  jd_skill_count          - Number of skills in JD
    [7]  matched_skill_count     - Count of matched skills
    [8]  missing_skill_count     - Count of missing skills
    [9]  log_resume_word_count   - log1p(resume length in words)
    [10] log_jd_word_count       - log1p(JD length in words)
    [11] length_ratio            - Resume/JD length ratio (raw words)

    Removed in Phase 2 (were always 0.5 during training):
      education_match, experience_match
    """
    log_resume = np.log1p(resume_word_count)
    log_jd = np.log1p(jd_word_count)
    length_ratio = resume_word_count / max(jd_word_count, 1)

    return np.array([
        tfidf_sim,
        sbert_sim,
        skill_coverage,
        required_skill_coverage,
        preferred_skill_coverage,
        float(resume_skill_count),
        float(jd_skill_count),
        float(matched_skill_count),
        float(missing_skill_count),
        log_resume,
        log_jd,
        length_ratio,
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
    "log_resume_word_count",
    "log_jd_word_count",
    "length_ratio",
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
    Train Logistic Regression (and XGBoost if available) on the provided
    feature arrays and return metrics plus trained model objects.

    Phase 2 fix: XGBoost now receives sample_weight computed from
    compute_sample_weight("balanced", y_train), putting it on an equal
    footing with Logistic Regression's class_weight="balanced".

    Returns dict of model results with metrics and trained models.
    """
    # Scale features — scaler fitted ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sample weights for models that accept them at .fit() time
    sample_weights = compute_sample_weight("balanced", y_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs", n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )

    results = {}

    print("\n" + "=" * 70)
    for name, model in models.items():
        print(f"\nTraining: {name}")
        if name == "XGBoost":
            # Phase 2 fix: pass sample_weight to XGBoost .fit()
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, preds)
        prec_macro = precision_score(y_test, preds, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, preds, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, preds)

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

        print(f"  Accuracy       : {acc:.4f}")
        print(f"  Macro Precision: {prec_macro:.4f}")
        print(f"  Macro Recall   : {rec_macro:.4f}")
        print(f"  Macro F1       : {f1_macro:.4f}")
        print(f"  Weighted F1    : {f1_weighted:.4f}")

    print("=" * 70)

    # Select best model by Macro F1 (primary metric)
    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    print(f"\nBest model: {best_name} (Macro F1={results[best_name]['f1_macro']:.4f})")

    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": results[best_name]["model"],
        "scaler": scaler,
        "label_encoder": label_encoder,
    }
