# Model Evaluation Results

## Model Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|-------|----------|-----------------|--------------|----------|-------------|
| Logistic Regression | 0.5642 | 0.5578 | 0.5642 | 0.5552 | 0.5552 |
| XGBoost | 0.5604 | 0.5578 | 0.5604 | 0.5545 | 0.5545 |

## Best Model: Logistic Regression

Primary Metric (Macro F1): **0.5552**

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Good Fit | 0.6168 | 0.5513 | 0.5822 | 800 |
| No Fit | 0.5880 | 0.7600 | 0.6630 | 800 |
| Potential Fit | 0.4685 | 0.3812 | 0.4204 | 800 |

## Natural Class Distribution Evaluation (N=1,200)

Evaluated on a stratified test sample reflecting the dataset's natural distribution (61.7% No Fit, 26.1% Potential Fit, 12.2% Good Fit):

- **Accuracy**: **63.33%**
- **Weighted F1**: **0.6374**
- **Macro F1**: **0.5365**
- **Macro Precision**: **0.5283**
- **Macro Recall**: **0.5641**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Good Fit | 0.3574 | 0.5753 | 0.4409 | 146 |
| No Fit | 0.8037 | 0.7635 | 0.7831 | 740 |
| Potential Fit | 0.4237 | 0.3535 | 0.3854 | 314 |