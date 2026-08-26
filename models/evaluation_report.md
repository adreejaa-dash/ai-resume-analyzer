# Model Evaluation Results

## Model Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|-------|----------|-----------------|--------------|----------|-------------|
| Logistic Regression | 0.5546 | 0.5448 | 0.5546 | 0.5446 | 0.5446 |
| XGBoost | 0.5458 | 0.5460 | 0.5458 | 0.5407 | 0.5407 |

## Best Model: Logistic Regression

Primary Metric (Macro F1): **0.5446**

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Good Fit | 0.5774 | 0.5825 | 0.5800 | 800 |
| No Fit | 0.5885 | 0.7275 | 0.6506 | 800 |
| Potential Fit | 0.4685 | 0.3538 | 0.4031 | 800 |