from flask import Flask, jsonify
from ml.pipeline import run_pipeline

app = Flask(__name__)

@app.route("/")
def home():
    return "ProximaScale is running"

@app.route("/predict")
def predict():
    prediction, decision = run_pipeline()
    return jsonify({
        "predicted_cpu": round(prediction, 2),
        "scaling_decision": decision
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
