"""
app.py — Flask load-generator + metrics endpoint for ProximaScale.
Person A owns this file.

Routes:
  GET  /          → light CPU load (math loop)
  GET  /heavy     → heavy CPU load
  GET  /metrics   → exposes request_rate for collector.py to poll
  GET  /health    → liveness check
  POST /predict   → (optional) calls Person B's model directly
"""
from flask import Flask, request, jsonify
import threading
import time
import math
import sys
import os

# Allow imports from project root (needed for /predict route)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)

# ── Thread-safe request counter ──────────────────────────────────────────────
_lock = threading.Lock()
_request_count = 0
_request_rate  = 0          # updated every 60 s by the background thread


@app.before_request
def count_request():
    global _request_count
    with _lock:
        _request_count += 1


def _reset_counter():
    """Background thread: slides the 60-second request-rate window."""
    global _request_count, _request_rate
    while True:
        time.sleep(60)
        with _lock:
            _request_rate  = _request_count
            _request_count = 0


threading.Thread(target=_reset_counter, daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Light CPU load — used by Locust normal_load scenario."""
    result = 0
    for i in range(1, 500_000):
        result += math.sqrt(i)
    return "App is running"


@app.route("/heavy")
def heavy():
    """Heavy CPU load — used by Locust spike scenario."""
    result = 0
    for i in range(1, 2_000_000):
        result += math.sqrt(i)
    return "Heavy load complete"


@app.route("/metrics")
def metrics():
    """Exposes current request_rate for monitoring/collector.py to poll."""
    with _lock:
        rate = _request_rate
    return jsonify({"request_rate": rate})


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
    app.run(host="0.0.0.0", port=5000, debug=False)
