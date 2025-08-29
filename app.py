# app.py
from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
artifact = joblib.load("model.joblib")
model = artifact["pipeline"]
labels = artifact["labels"]

@app.route("/", methods=["GET"])
def home():
    return "<h1>Welcome! This is the Root Page of My ML API.</h1>"

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        # Expect 4 Iris features
        features = [
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"],
        ]
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    X = np.array([features])
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]

    return jsonify({
        "predicted_class": pred_label,
        "probabilities": {labels[i]: float(p) for i, p in enumerate(probs)},
        "input": {
            "sepal_length": features[0],
            "sepal_width": features[1],
            "petal_length": features[2],
            "petal_width": features[3],
        }
    })

if __name__ == "__main__":
    # 0.0.0.0 makes it reachable from Docker
    app.run(host="0.0.0.0", port=5000)
