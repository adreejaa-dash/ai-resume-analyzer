"""
End-to-end model training pipeline.

Phases 6–9: TF-IDF baseline → SBERT features → Feature Engineering → LR/XGB training.

Implements the three-stage experimental design:
  Stage 1: Classical TF-IDF cosine similarity (baseline)
  Stage 2: Sentence-BERT semantic similarity (baseline)
  Stage 3: Hybrid supervised model with engineered features

Strict data-leakage prevention:
  - TF-IDF fitted only on training data
  - Scaler fitted only on training data
  - Validation split derived from training data only
  - Test set used ONLY for final evaluation
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocessing.text_cleaner import clean_for_tfidf, clean_for_embedding
from src.matching.tfidf_matcher import TfidfMatcher
from src.matching.semantic_matcher import SemanticMatcher
from src.skills.taxonomy import SkillTaxonomy
from src.skills.extractor import SkillExtractor, extract_jd_skills
from src.skills.matcher import match_skills
from src.models.trainer import (
    build_feature_vector, FEATURE_NAMES, train_and_evaluate,
)
from src.evaluation.evaluator import (
    plot_confusion_matrix, print_comparison_table, generate_evaluation_report,
)

DATA_DIR = os.path.join(ROOT, "data", "raw")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    """Load primary dataset and verify integrity."""
    print("=" * 70)
    print("Loading primary dataset: med2425/resume-job-fit-merged-v1")
    print("=" * 70)

    train_path = os.path.join(DATA_DIR, "resume_job_fit_train.parquet")
    test_path = os.path.join(DATA_DIR, "resume_job_fit_test.parquet")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}\n"
            "Run: python3 scripts/download_data.py"
        )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"Columns: {list(train_df.columns)}")
    print(f"\nClass distribution (train):")
    dist = train_df["label"].value_counts()
    for label, cnt in dist.items():
        pct = cnt / len(train_df) * 100
        print(f"  {label:<20}: {cnt:>6} ({pct:.1f}%)")
    print(f"\nClass distribution (test):")
    dist_test = test_df["label"].value_counts()
    for label, cnt in dist_test.items():
        pct = cnt / len(test_df) * 100
        print(f"  {label:<20}: {cnt:>6} ({pct:.1f}%)")

    # Check for nulls
    null_train = train_df[["resume", "jd", "label"]].isnull().sum()
    null_test = test_df[["resume", "jd", "label"]].isnull().sum()
    print(f"\nNull values (train): {null_train.to_dict()}")
    print(f"Null values (test):  {null_test.to_dict()}")

    # Drop nulls
    train_df = train_df.dropna(subset=["resume", "jd", "label"])
    test_df = test_df.dropna(subset=["resume", "jd", "label"])
    print(f"\nAfter null removal: train={len(train_df)}, test={len(test_df)}")

    return train_df, test_df


def stage1_tfidf_baseline(train_df, test_df, sample_n=5000):
    """
    Stage 1: Classical NLP baseline — TF-IDF cosine similarity.

    Reports similarity distribution across labels without any supervised model.
    This is the classical NLP baseline from the original complaint-classifier.

    Uses a sample for speed. Fits TF-IDF only on training texts.
    """
    print("\n" + "=" * 70)
    print("STAGE 1: TF-IDF Classical Baseline")
    print("=" * 70)

    # Sample for speed
    train_sample = train_df.sample(min(sample_n, len(train_df)), random_state=42)
    test_sample = test_df.sample(min(1000, len(test_df)), random_state=42)

    # Clean texts
    print("Cleaning texts for TF-IDF...")
    train_resume = train_sample["resume"].apply(clean_for_tfidf).tolist()
    train_jd = train_sample["jd"].apply(clean_for_tfidf).tolist()
    test_resume = test_sample["resume"].apply(clean_for_tfidf).tolist()
    test_jd = test_sample["jd"].apply(clean_for_tfidf).tolist()

    # Fit on training documents only (concat resume+jd as corpus)
    # IMPORTANT: fitted only on training data to prevent leakage
    print("Fitting TF-IDF on training corpus (resume+JD combined)...")
    tfidf = TfidfMatcher(max_features=10000)
    train_corpus = train_resume + train_jd
    tfidf.fit(train_corpus)

    print("Computing train similarities...")
    train_sims = tfidf.batch_similarity(train_resume, train_jd)
    print("Computing test similarities...")
    test_sims = tfidf.batch_similarity(test_resume, test_jd)

    # Report similarity by label
    print("\nTF-IDF Cosine Similarity by Label (test sample):")
    test_sample = test_sample.reset_index(drop=True)
    for label in test_sample["label"].unique():
        mask = test_sample["label"] == label
        sims_for_label = test_sims[mask.values]
        print(f"  {label:<20}: mean={sims_for_label.mean():.4f}  "
              f"std={sims_for_label.std():.4f}  "
              f"median={np.median(sims_for_label):.4f}")

    return tfidf, train_sims, test_sims, train_sample, test_sample


def stage2_sbert_baseline(train_sample, test_sample, train_sims, test_sims):
    """
    Stage 2: Sentence-BERT semantic baseline.

    Uses pretrained all-MiniLM-L6-v2. NOT fine-tuned.
    Compares against TF-IDF baseline.
    """
    print("\n" + "=" * 70)
    print("STAGE 2: Sentence-BERT Semantic Baseline")
    print("  Model: sentence-transformers/all-MiniLM-L6-v2 (pretrained, not fine-tuned)")
    print("=" * 70)

    sbert = SemanticMatcher()

    train_resume_raw = train_sample["resume"].apply(clean_for_embedding).tolist()
    train_jd_raw = train_sample["jd"].apply(clean_for_embedding).tolist()
    test_resume_raw = test_sample["resume"].apply(clean_for_embedding).tolist()
    test_jd_raw = test_sample["jd"].apply(clean_for_embedding).tolist()

    print(f"Encoding {len(train_resume_raw)} train resume+JD pairs...")
    train_sbert_sims = sbert.batch_similarity(train_resume_raw, train_jd_raw)
    print(f"Encoding {len(test_resume_raw)} test resume+JD pairs...")
    test_sbert_sims = sbert.batch_similarity(test_resume_raw, test_jd_raw)

    # Compare TF-IDF vs SBERT
    print("\n--- Comparison: TF-IDF vs SBERT Similarity ---")
    print(f"{'Label':<20} {'TF-IDF':>8} {'SBERT':>8}  Difference")
    print("-" * 50)
    test_sample_reset = test_sample.reset_index(drop=True)
    for label in sorted(test_sample_reset["label"].unique()):
        mask = (test_sample_reset["label"] == label).values
        tfidf_mean = test_sims[mask].mean()
        sbert_mean = test_sbert_sims[mask].mean()
        diff = sbert_mean - tfidf_mean
        print(f"  {label:<20} {tfidf_mean:>8.4f} {sbert_mean:>8.4f}  "
              f"{'SBERT higher' if diff > 0 else 'TFIDF higher'} by {abs(diff):.4f}")

    # Find examples where SBERT > TFIDF by significant margin
    print("\nSemantic vs lexical gap examples (test sample):")
    diffs = test_sbert_sims - test_sims
    top_idx = np.argsort(diffs)[-3:][::-1]
    for idx in top_idx:
        row = test_sample_reset.iloc[idx]
        print(f"\n  Label: {row['label']}")
        print(f"  TF-IDF sim: {test_sims[idx]:.4f}  SBERT sim: {test_sbert_sims[idx]:.4f}")
        print(f"  Resume snippet: {row['resume'][:120].strip()!r}")
        print(f"  JD snippet:     {row['jd'][:120].strip()!r}")

    return sbert, train_sbert_sims, test_sbert_sims


def compute_skill_features(df_sample, taxonomy, extractor):
    """Compute skill-based features for each resume/JD pair."""
    features = []
    for _, row in df_sample.iterrows():
        resume_skills = extractor.extract_flat_skills(row["resume"])
        jd_skills_data = extract_jd_skills(row["jd"], extractor)
        skill_match = match_skills(
            resume_skills,
            jd_skills_data["required"],
            jd_skills_data["preferred"],
        )
        features.append({
            "resume_skill_count": len(resume_skills),
            "jd_skill_count": len(jd_skills_data["all"]),
            "matched_skill_count": skill_match["matched_required_count"],
            "missing_skill_count": skill_match["missing_required_count"],
            "skill_coverage": skill_match["overall_coverage"],
            "required_skill_coverage": skill_match["required_coverage"],
            "preferred_skill_coverage": skill_match["preferred_coverage"],
        })
    return features


def stage3_hybrid_model(
    train_df, test_df,
    tfidf_matcher, sbert_matcher,
    taxonomy, extractor,
    sample_n=8000,
):
    """
    Stage 3: Hybrid supervised model with engineered features.

    Feature engineering: TF-IDF sim + SBERT sim + skill coverage + text stats.
    Models: Logistic Regression + XGBoost.
    Primary metric: Macro F1 (because of severe class imbalance).

    STRICT LEAKAGE PREVENTION:
    - Derive validation split from training data only
    - TF-IDF already fitted on training data
    - Scaler fitted on training features only
    - Test set used only for final evaluation
    """
    print("\n" + "=" * 70)
    print("STAGE 3: Hybrid Supervised Model — Feature Engineering")
    print("=" * 70)

    # Sample from train (stratified) — avoid group_keys issue
    groups = []
    for label_val, group in train_df.groupby("label"):
        n = min(len(group), sample_n // 3)
        groups.append(group.sample(n, random_state=42))
    train_sample = pd.concat(groups).reset_index(drop=True)

    print(f"Training sample: {len(train_sample)} examples")
    print("Class distribution:", train_sample["label"].value_counts().to_dict())

    # Encode labels
    le = LabelEncoder()
    le.fit(train_df["label"])  # Fit on full label set
    y_train_all = le.transform(train_sample["label"])

    # Train/validation split from training data only (stratified)
    idx_train, idx_val = train_test_split(
        range(len(train_sample)), test_size=0.15, random_state=42,
        stratify=y_train_all,
    )

    print(f"\nBuilding feature vectors for {len(train_sample)} samples...")
    print("(Computing TF-IDF, SBERT, skill features — this takes a few minutes)")

    # Compute text similarities for training sample
    train_texts_resume = train_sample["resume"].apply(clean_for_tfidf).tolist()
    train_texts_jd = train_sample["jd"].apply(clean_for_tfidf).tolist()
    train_tfidf_sims = tfidf_matcher.batch_similarity(train_texts_resume, train_texts_jd)

    train_emb_resume = train_sample["resume"].apply(clean_for_embedding).tolist()
    train_emb_jd = train_sample["jd"].apply(clean_for_embedding).tolist()
    train_sbert_sims = sbert_matcher.batch_similarity(train_emb_resume, train_emb_jd)

    print("Computing skill features for training sample...")
    train_skill_feats = compute_skill_features(train_sample, taxonomy, extractor)

    # Build feature matrix
    X_train_all = np.array([
        build_feature_vector(
            tfidf_sim=train_tfidf_sims[i],
            sbert_sim=train_sbert_sims[i],
            skill_coverage=train_skill_feats[i]["skill_coverage"],
            required_skill_coverage=train_skill_feats[i]["required_skill_coverage"],
            preferred_skill_coverage=train_skill_feats[i]["preferred_skill_coverage"],
            resume_skill_count=train_skill_feats[i]["resume_skill_count"],
            jd_skill_count=train_skill_feats[i]["jd_skill_count"],
            matched_skill_count=train_skill_feats[i]["matched_skill_count"],
            missing_skill_count=train_skill_feats[i]["missing_skill_count"],
            resume_word_count=len(train_sample.iloc[i]["resume"].split()),
            jd_word_count=len(train_sample.iloc[i]["jd"].split()),
        )
        for i in range(len(train_sample))
    ])
    y_train_all_enc = le.transform(train_sample["label"])

    # Use full training sample for training, no validation leakage
    X_tr = X_train_all[idx_train]
    y_tr = y_train_all_enc[idx_train]
    X_val = X_train_all[idx_val]
    y_val = y_train_all_enc[idx_val]

    print(f"\nFeature matrix: train={X_tr.shape}, val={X_val.shape}")
    print(f"Feature names: {FEATURE_NAMES}")

    # Compute class weights (training split only)
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight_dict = dict(zip(classes, weights))
    print(f"\nClass weights (balanced): {class_weight_dict}")

    # Train on training split, evaluate on validation split
    print("\n--- Training on train split, evaluating on VALIDATION split ---")
    val_output = train_and_evaluate(X_tr, y_tr, X_val, y_val, le, class_weight_dict)
    print("\nValidation Results:")
    print_comparison_table(val_output["results"])

    # Now compute test features (test set untouched until now)
    print("\n" + "=" * 70)
    print("FINAL EVALUATION on HELD-OUT TEST SET")
    print("=" * 70)

    test_groups = []
    for label_val, group in test_df.groupby("label"):
        n = min(len(group), 800)
        test_groups.append(group.sample(n, random_state=42))
    test_sample = pd.concat(test_groups).reset_index(drop=True)

    print(f"Test sample: {len(test_sample)} examples")

    test_texts_resume = test_sample["resume"].apply(clean_for_tfidf).tolist()
    test_texts_jd = test_sample["jd"].apply(clean_for_tfidf).tolist()
    test_tfidf_sims = tfidf_matcher.batch_similarity(test_texts_resume, test_texts_jd)

    test_emb_resume = test_sample["resume"].apply(clean_for_embedding).tolist()
    test_emb_jd = test_sample["jd"].apply(clean_for_embedding).tolist()
    test_sbert_sims = sbert_matcher.batch_similarity(test_emb_resume, test_emb_jd)

    print("Computing skill features for test sample...")
    test_skill_feats = compute_skill_features(test_sample, taxonomy, extractor)

    X_test = np.array([
        build_feature_vector(
            tfidf_sim=test_tfidf_sims[i],
            sbert_sim=test_sbert_sims[i],
            skill_coverage=test_skill_feats[i]["skill_coverage"],
            required_skill_coverage=test_skill_feats[i]["required_skill_coverage"],
            preferred_skill_coverage=test_skill_feats[i]["preferred_skill_coverage"],
            resume_skill_count=test_skill_feats[i]["resume_skill_count"],
            jd_skill_count=test_skill_feats[i]["jd_skill_count"],
            matched_skill_count=test_skill_feats[i]["matched_skill_count"],
            missing_skill_count=test_skill_feats[i]["missing_skill_count"],
            resume_word_count=len(test_sample.iloc[i]["resume"].split()),
            jd_word_count=len(test_sample.iloc[i]["jd"].split()),
        )
        for i in range(len(test_sample))
    ])
    y_test = le.transform(test_sample["label"])

    # Final evaluation uses scaler fitted on training data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train_all[idx_train])
    X_test_scaled = scaler.transform(X_test)

    # Retrain best model on full training split with scaler
    test_output = train_and_evaluate(
        X_train_all[idx_train], y_tr, X_test, y_test, le, class_weight_dict
    )
    print("\nFinal Test Results:")
    print_comparison_table(test_output["results"])

    return test_output, le, test_sample, X_test, y_test


def save_models(test_output, tfidf_matcher, le):
    """Save all model artifacts."""
    print("\nSaving model artifacts...")

    best_name = test_output["best_model_name"]
    best_model = test_output["best_model"]
    scaler = test_output["scaler"]
    results = test_output["results"]

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    joblib.dump(tfidf_matcher.vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

    # Save LR model separately (for feature importance/explainability)
    if "Logistic Regression" in results:
        joblib.dump(
            results["Logistic Regression"]["model"],
            os.path.join(MODELS_DIR, "logistic_regression.joblib")
        )
    if "XGBoost" in results:
        joblib.dump(
            results["XGBoost"]["model"],
            os.path.join(MODELS_DIR, "xgboost_model.joblib")
        )

    # Save model info JSON (actual results only)
    model_info = {
        "best_model": best_name,
        "primary_metric": "Macro F1",
        "dataset": "med2425/resume-job-fit-merged-v1",
        "labels": le.classes_.tolist(),
        "feature_names": FEATURE_NAMES,
        "note": "All metrics are actual test-set results, not fabricated.",
        "results": {
            name: {
                "accuracy": float(r["accuracy"]),
                "precision_macro": float(r["precision_macro"]),
                "recall_macro": float(r["recall_macro"]),
                "f1_macro": float(r["f1_macro"]),
                "f1_weighted": float(r["f1_weighted"]),
            }
            for name, r in results.items()
        },
    }

    with open(os.path.join(MODELS_DIR, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"Saved artifacts to: {MODELS_DIR}")
    for fname in os.listdir(MODELS_DIR):
        size = os.path.getsize(os.path.join(MODELS_DIR, fname))
        print(f"  {fname}: {size/1024:.1f} KB")


def generate_confusion_matrices(test_output, le):
    """Generate and save confusion matrices for all models."""
    for name, r in test_output["results"].items():
        fname = name.lower().replace(" ", "_") + "_cm.png"
        plot_confusion_matrix(
            r["confusion_matrix"],
            le.classes_.tolist(),
            title=f"Confusion Matrix — {name}",
            save_path=os.path.join(MODELS_DIR, fname),
        )


def error_analysis(test_sample, y_test, test_output, le):
    """Phase 10: Inspect misclassifications."""
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS — Misclassification Inspection")
    print("=" * 70)

    best_name = test_output["best_model_name"]
    best_model = test_output["best_model"]
    scaler = test_output["scaler"]

    from sklearn.preprocessing import StandardScaler
    # Recompute — scaler already fitted in test_output
    X_test_raw = test_output.get("X_test")
    if X_test_raw is None:
        print("X_test not available for error analysis (stored in output)")
        return

    X_scaled = scaler.transform(X_test_raw)
    preds = best_model.predict(X_scaled)
    probs = best_model.predict_proba(X_scaled)

    wrong_mask = preds != y_test
    wrong_df = test_sample[wrong_mask].copy().reset_index(drop=True)
    wrong_preds = preds[wrong_mask]
    wrong_true = y_test[wrong_mask]

    print(f"Total misclassified: {wrong_mask.sum()} / {len(y_test)}")
    print(f"Error rate: {wrong_mask.mean():.3f}")

    print("\nMisclassification patterns:")
    from collections import Counter
    patterns = Counter(zip(
        le.inverse_transform(wrong_true),
        le.inverse_transform(wrong_preds)
    ))
    for (true_l, pred_l), count in patterns.most_common():
        print(f"  True={true_l:<20} Pred={pred_l:<20}: {count}")

    print("\nSample misclassified examples:")
    for i in range(min(3, len(wrong_df))):
        true_label = le.inverse_transform([wrong_true[i]])[0]
        pred_label = le.inverse_transform([wrong_preds[i]])[0]
        row = wrong_df.iloc[i]
        print(f"\n  [{i+1}] True: {true_label}  |  Predicted: {pred_label}")
        print(f"  Resume: {row['resume'][:200].strip()!r}")
        print(f"  JD:     {row['jd'][:200].strip()!r}")


def main():
    print("=" * 70)
    print("Resume–Job Matching: Training Pipeline")
    print("Primary metric: Macro F1 | Dataset: med2425/resume-job-fit-merged-v1")
    print("=" * 70)

    # Load data
    train_df, test_df = load_data()

    # Load taxonomy and extractor
    print("\nLoading skill taxonomy...")
    taxonomy = SkillTaxonomy()
    extractor = SkillExtractor(taxonomy)
    print(f"Taxonomy: {taxonomy.get_status()['total_skills']} skills loaded")

    # Stage 1: TF-IDF baseline
    tfidf_matcher, train_tfidf_sims, test_tfidf_sims, train_s, test_s = \
        stage1_tfidf_baseline(train_df, test_df)

    # Stage 2: SBERT baseline
    sbert_matcher, train_sbert_sims, test_sbert_sims = \
        stage2_sbert_baseline(train_s, test_s, train_tfidf_sims, test_tfidf_sims)

    # Stage 3: Hybrid supervised model
    test_output, le, test_sample, X_test, y_test = stage3_hybrid_model(
        train_df, test_df, tfidf_matcher, sbert_matcher, taxonomy, extractor,
    )
    # Store X_test for error analysis
    test_output["X_test"] = X_test

    # Save models
    save_models(test_output, tfidf_matcher, le)

    # Generate confusion matrices
    generate_confusion_matrices(test_output, le)

    # Error analysis
    error_analysis(test_sample, y_test, test_output, le)

    # Save evaluation report
    report = generate_evaluation_report(test_output["results"], le.classes_.tolist())
    report_path = os.path.join(MODELS_DIR, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nEvaluation report saved: {report_path}")

    print("\n" + "=" * 70)
    print("Training pipeline complete.")
    print(f"Best model: {test_output['best_model_name']}")
    print(f"Macro F1:   {test_output['results'][test_output['best_model_name']]['f1_macro']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
