"""
Model evaluation utilities.

Generates confusion matrices, comparison tables, and evaluation reports.
Reuses the evaluation pattern from the original complaint-classifier.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Dict, List
import os


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    save_path: str = None,
):
    """
    Plot and optionally save a confusion matrix.
    Reuses the plotting pattern from the original complaint-classifier.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved: {save_path}")
    
    plt.close()


def print_comparison_table(results: Dict):
    """
    Print a model comparison table.
    Same format as the original complaint-classifier.
    """
    print(f"\n{'Model':<30} {'Acc':>6} {'P(M)':>6} {'R(M)':>6} {'F1(M)':>6} {'F1(W)':>6}")
    print("-" * 72)
    for name, r in results.items():
        print(
            f"{name:<30} "
            f"{r['accuracy']:>6.4f} "
            f"{r['precision_macro']:>6.4f} "
            f"{r['recall_macro']:>6.4f} "
            f"{r['f1_macro']:>6.4f} "
            f"{r['f1_weighted']:>6.4f}"
        )


def generate_evaluation_report(results: Dict, class_names: list) -> str:
    """Generate a markdown-formatted evaluation report."""
    lines = ["# Model Evaluation Results\n"]
    lines.append("## Model Comparison\n")
    lines.append("| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |")
    lines.append("|-------|----------|-----------------|--------------|----------|-------------|")
    
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['accuracy']:.4f} | "
            f"{r['precision_macro']:.4f} | "
            f"{r['recall_macro']:.4f} | "
            f"{r['f1_macro']:.4f} | "
            f"{r['f1_weighted']:.4f} |"
        )
    
    # Per-class metrics for best model
    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    lines.append(f"\n## Best Model: {best_name}\n")
    lines.append(f"Primary Metric (Macro F1): **{results[best_name]['f1_macro']:.4f}**\n")
    
    report = results[best_name].get("classification_report", {})
    if report:
        lines.append("### Per-Class Metrics\n")
        lines.append("| Class | Precision | Recall | F1-Score | Support |")
        lines.append("|-------|-----------|--------|----------|---------|")
        for cls in class_names:
            if cls in report:
                m = report[cls]
                lines.append(
                    f"| {cls} | {m['precision']:.4f} | "
                    f"{m['recall']:.4f} | {m['f1-score']:.4f} | "
                    f"{int(m['support'])} |"
                )
    
    return "\n".join(lines)
