from flask import Flask, jsonify
import threading
import time
import math

app = Flask(__name__)

# Thread-safe request counter
_lock = threading.Lock()
_request_count = 0
_request_rate = 0  # requests in the last 60s window, updated every 60s

@app.before_request
def count_request():
    global _request_count
    with _lock:
        _request_count += 1

def _reset_counter():
    """Every 60s, snapshot count into request_rate, then reset."""
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
    # CPU-intensive: simulate real load
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
