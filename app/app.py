from flask import Flask, request, jsonify
import threading
import time
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.predict import predict

app = Flask(__name__)

# Thread-safe request counter (Person A)
_lock = threading.Lock()
_request_count = 0
_request_rate = 0

@app.before_request
def count_request():
    global _request_count
    with _lock:
        _request_count += 1

def _reset_counter():
    global _request_count, _request_rate
    while True:
        time.sleep(60)
        with _lock:
            _request_rate = _request_count
            _request_count = 0

threading.Thread(target=_reset_counter, daemon=True).start()

@app.route("/")
def home():
    result = 0
    for i in range(1, 500000):
        result += math.sqrt(i)
    return "App is running"

@app.route("/heavy")
def heavy():
    result = 0
    for i in range(1, 2000000):
        result += math.sqrt(i)
    return "Heavy load complete"

@app.route("/metrics")
def metrics():
    """Exposes current request_rate for collector.py to poll."""
    with _lock:
        rate = _request_rate
    return jsonify({"request_rate": rate})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ProximaScale API is running!"})

@app.route("/predict", methods=["POST"])
def get_prediction():
    """
    Accepts 10 metric records and returns CPU predictions + anomaly flag.
    """
    data = request.get_json()

    if not data or "records" not in data:
        return jsonify({"error": "Missing records in request"}), 400

    if len(data["records"]) != 10:
        return jsonify({"error": "Exactly 10 records required"}), 400

    result = predict(data["records"])
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
