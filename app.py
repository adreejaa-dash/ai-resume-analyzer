import os
import joblib
from flask import Flask, render_template, request
from preprocess import clean_text

app = Flask(__name__)

print("Loading models, vectorizer, and label encoder...")
model      = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
model_info = joblib.load("model_info.pkl")
le         = joblib.load("label_encoder.pkl")
model_name = model_info["name"]
model_acc  = model_info["accuracy"]

# Load LR model for top-word explanations (has coef_), fallback to best model
lr_model = joblib.load("lr_model.pkl") if os.path.exists("lr_model.pkl") else model

TOP_WORDS = 10


def get_top_words(text_vec, predicted_class):
    if not hasattr(lr_model, "coef_"):
        return []
    feature_names = vectorizer.get_feature_names_out()
    
    # In multi-class LogisticRegression, coef_ is (n_classes, n_features)
    # If binary, it's (1, n_features). Handle both gracefully:
    if lr_model.coef_.shape[0] > 1:
        coef = lr_model.coef_[predicted_class]
    else:
        coef = lr_model.coef_[0]
        if predicted_class == 0:
            coef = -coef

    scores = text_vec.toarray()[0] * coef
    top_indices = scores.argsort()[-TOP_WORDS:][::-1]
    return [(feature_names[i], round(float(scores[i]), 4)) for i in top_indices if scores[i] > 0]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    
    if request.method == "POST":
        complaint_text = request.form.get("complaint_text", "").strip()
        
        if not complaint_text:
            error = "Please enter a complaint."
        elif len(complaint_text) < 15:
            error = "Complaint is too short. Please provide more details."
        else:
            processed = clean_text(complaint_text)
            
            # Additional check: what if preprocessing removes all words?
            if not processed:
                error = "Could not extract meaningful words from the complaint."
            else:
                vec = vectorizer.transform([processed])
                prediction = model.predict(vec)[0]
                proba = model.predict_proba(vec)[0]
                confidence = round(float(proba[prediction]) * 100, 2)
                
                label = le.inverse_transform([prediction])[0]
                top_words = get_top_words(vec, prediction)
                
                # Suggested Routing Map
                routing_map = {
                    "Order Cancellation": "Orders Team",
                    "Order Not Received / Shipping Delay": "Shipping & Fulfillment",
                    "Billing / Payment Issue": "Billing Department",
                    "Refund / Return Request": "Returns & Refunds",
                    "Damaged / Defective Product": "Returns & Refunds",
                    "Wrong Item Delivered": "Returns & Refunds",
                    "Account / Login Issue": "Technical Support",
                    "Product Quality / Not as Described": "Quality Assurance",
                    "Customer Service Complaint": "Escalation Team"
                }
                suggested_routing = routing_map.get(label, "General Support")
                
                result = {
                    "label": label,
                    "confidence": confidence,
                    "model": model_name,
                    "model_accuracy": round(model_acc * 100, 2),
                    "top_words": top_words,
                    "routing": suggested_routing
                }
    
    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
