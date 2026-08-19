# Model Evaluation Results

## Model Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|-------|----------|-----------------|--------------|----------|-------------|
| Logistic Regression | 0.5546 | 0.5448 | 0.5546 | 0.5446 | 0.5446 |
| Gaussian Naive Bayes | 0.5617 | 0.5491 | 0.5617 | 0.5449 | 0.5449 |
| XGBoost | 0.5458 | 0.5460 | 0.5458 | 0.5407 | 0.5407 |

## Best Model: Gaussian Naive Bayes

Primary Metric (Macro F1): **0.5449**

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Good Fit | 0.6041 | 0.5913 | 0.5976 | 800 |
| No Fit | 0.5788 | 0.7850 | 0.6663 | 800 |
| Potential Fit | 0.4643 | 0.3088 | 0.3709 | 800 |