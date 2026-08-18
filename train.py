import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    print(f"[WARN] XGBoost not available ({e}). Using RandomForest instead.")
    XGBOOST_AVAILABLE = False

from preprocess import clean_text

print("Loading data...")
df = pd.read_csv("data/hf_customer_support.csv", usecols=["instruction", "intent"])
df = df.dropna()
df.rename(columns={"instruction": "text", "intent": "label"}, inplace=True)

INTENT_MAPPING = {
    'cancel_order': 'Order Cancellation',
    'change_order': 'Order Cancellation',
    'change_shipping_address': 'Order Not Received / Shipping Delay',
    'check_cancellation_fee': 'Order Cancellation',
    'check_invoice': 'Billing / Payment Issue',
    'check_payment_methods': 'Billing / Payment Issue',
    'check_refund_policy': 'Refund / Return Request',
    'complaint': 'Damaged / Defective Product',
    'contact_customer_service': 'Customer Service Complaint',
    'contact_human_agent': 'Customer Service Complaint',
    'create_account': 'Account / Login Issue',
    'delete_account': 'Account / Login Issue',
    'delivery_options': 'Order Not Received / Shipping Delay',
    'delivery_period': 'Order Not Received / Shipping Delay',
    'edit_account': 'Account / Login Issue',
    'get_invoice': 'Billing / Payment Issue',
    'get_refund': 'Refund / Return Request',
    'newsletter_subscription': 'Account / Login Issue',
    'payment_issue': 'Billing / Payment Issue',
    'place_order': 'Wrong Item Delivered',
    'recover_password': 'Account / Login Issue',
    'registration_problems': 'Account / Login Issue',
    'review': 'Product Quality / Not as Described',
    'set_up_shipping_address': 'Order Not Received / Shipping Delay',
    'switch_account': 'Account / Login Issue',
    'track_order': 'Order Not Received / Shipping Delay',
    'track_refund': 'Refund / Return Request'
}
df['label'] = df['label'].map(INTENT_MAPPING)
df = df.dropna(subset=['label'])

# Drop categories with less than 200 examples
category_counts = df["label"].value_counts()
valid_categories = category_counts[category_counts >= 200].index
df = df[df["label"].isin(valid_categories)]

print(f"Dataset size: {len(df)} rows  |  Label distribution:\n{df['label'].value_counts().to_string()}")

print("Encoding labels...")
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

print("Preprocessing text (this may take a while)...")
df["processed"] = df["text"].apply(clean_text)

X = df["processed"]
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

if XGBOOST_AVAILABLE:
    third_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 eval_metric="mlogloss",
                                 tree_method="hist", n_jobs=-1, random_state=42)
    third_name = "XGBoost"
else:
    third_model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    third_name = "Random Forest"

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1),
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
    third_name: third_model,
}

results = {}

print("\n" + "="*70)
for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec  = recall_score(y_test, preds, average="macro", zero_division=0)
    f1   = f1_score(y_test, preds, average="macro", zero_division=0)
    cm   = confusion_matrix(y_test, preds)

    results[name] = {"model": model, "accuracy": acc, "precision": prec,
                     "recall": rec, "f1": f1, "cm": cm}

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1       : {f1:.4f}")
print("="*70)

best_name = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_name]["model"]
best_acc   = results[best_name]["accuracy"]
best_cm    = results[best_name]["cm"]
print(f"\nBest model: {best_name}  (F1={results[best_name]['f1']:.4f})")

print("Saving model, vectorizer, and label encoder...")
joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(results["Logistic Regression"]["model"], "lr_model.pkl")

model_info = {
    "name": best_name,
    "accuracy": best_acc,
    "all_results": {k: {m: v for m, v in r.items() if m not in ["model", "cm"]} for k, r in results.items()}
}
joblib.dump(model_info, "model_info.pkl")

print("Plotting confusion matrix for best model...")
plt.figure(figsize=(10, 8))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix ({best_name})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

print("Done! model.pkl, vectorizer.pkl, label_encoder.pkl, lr_model.pkl, model_info.pkl, and confusion_matrix.png saved.")
print("\nFull comparison:")
print(f"{'Model':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-"*58)
for name, r in results.items():
    print(f"{name:<30} {r['accuracy']:>6.4f} {r['precision']:>6.4f} {r['recall']:>6.4f} {r['f1']:>6.4f}")
