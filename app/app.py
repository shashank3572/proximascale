"""
app.py — Flask load-generator + metrics endpoint for ProximaScale.
Person A owns this file.

Routes:
  GET  /work/light  → light CPU load (used by Locust normal_load scenario)
  GET  /work/heavy  → heavy CPU load (used by Locust spike scenario)
  GET  /health      → liveness check
  POST /predict     → (optional) calls Person B's model directly

Request counting is done via monitoring.metrics (SQLite-backed counter),
not an in-process counter -- monitoring/collector.py reads it directly
with reset_request_count() rather than polling an HTTP endpoint, so it
works correctly even if collector.py runs in a separate process/container
from this Flask app.
"""
from flask import Flask, request, jsonify
import math
import sys
import os
from monitoring.metrics import increment_request_count

# Allow imports from project root (needed for /predict route)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)


@app.before_request
def count_request():
    increment_request_count()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/work/light")
def light_work():
    """Light CPU load — used by Locust normal_load scenario."""
    result = 0
    for i in range(200_000):
        result += math.sqrt(i)
    return "Light work completed"


@app.route("/work/heavy")
def heavy_work():
    """Heavy CPU load — used by Locust spike scenario."""
    result = 0
    for i in range(5_000_000):
        result += math.sqrt(i)
    return "Heavy work completed"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ProximaScale API is running!"})


@app.route("/predict", methods=["POST"])
def get_prediction():
    """
    (Optional - Semester 1 bonus) Accepts 10 metric records and returns
    CPU predictions + anomaly flag by calling Person B's model directly.

    FIX: Import is lazy (inside the function) so Flask starts even if
    TensorFlow or the .keras file are not available in this environment.
    Without this fix the entire app crashed on startup when run inside Docker
    (model/ directory is not copied into the container).
    """
    try:
        from model.predict import predict          # lazy import — safe
    except ImportError as e:
        return jsonify({
            "error": "ML model not available in this environment.",
            "detail": str(e)
        }), 503

    data = request.get_json()

    if not data or "records" not in data:
        return jsonify({"error": "Missing 'records' in request body"}), 400

    if len(data["records"]) != 10:
        return jsonify({"error": "Exactly 10 records required"}), 400

    result = predict(data["records"])
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)