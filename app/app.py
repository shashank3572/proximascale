"""
app.py — Flask load-generator + metrics endpoint for ProximaScale.
Person A owns this file.

Routes:
  GET  /            → landing page (status + endpoint list, for humans/panel)
  GET  /work/light  → light CPU load (used by Locust normal_load scenario)
  GET  /work/heavy  → heavy CPU load (used by Locust spike scenario)
  GET  /health      → liveness check
  GET  /request-rate → current (unreset) request counter, JSON -- polled by
                        the "/" landing page's live number, non-destructive
  POST /predict     → (optional) calls Person B's model directly

Request counting is done via monitoring.metrics (SQLite-backed counter),
not an in-process counter -- monitoring/collector.py reads it directly
with reset_request_count() rather than polling an HTTP endpoint, so it
works correctly even if collector.py runs in a separate process/container
from this Flask app.

NOTE: "/", "/health", "/request-rate" and "/predict" are deliberately
excluded from the request counter below. They aren't load-test traffic --
if they counted, someone refreshing the landing page (or its own
auto-polling JS) would inflate monitoring/collector.py's request_rate
feature, which the residual-stacking model actually trains and predicts
on. Only /work/light and /work/heavy (the real Locust routes) count.
"""
from flask import Flask, request, jsonify, Response
import math
import sys
import os
from monitoring.metrics import increment_request_count, get_request_count

# Allow imports from project root (needed for /predict route)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)

_EXCLUDED_FROM_COUNT = {"/", "/health", "/request-rate", "/predict", "/favicon.ico"}


@app.before_request
def count_request():
    if request.path not in _EXCLUDED_FROM_COUNT:
        increment_request_count()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def landing_page():
    """Simple demo landing page — mirrors the pre-integration reference
    screenshot. Uses get_request_count() (read-only) rather than
    reset_request_count(), so viewing this page never clears the counter
    monitoring/collector.py is relying on."""
    count = get_request_count()
    return f"""
    <html>
    <head><title>ProximaScale</title></head>
    <body style="font-family: sans-serif; max-width: 700px; margin: 40px auto;">
        <h1>ProximaScale Flask App Running!</h1>
        <hr>
        <div style="border:1px solid #ddd; border-radius:8px; padding:16px; margin:16px 0;">
            <h3>📈 Current Request Count</h3>
            <p style="font-size:28px; color:#2563eb;"><b>{count}</b></p>
            <p style="color:#666; font-size:13px;">
                Since last collector poll (resets every ~30s when
                monitoring/collector.py is running).
            </p>
        </div>
        <div style="border:1px solid #ddd; border-radius:8px; padding:16px;">
            <h3>🔗 Available Endpoints</h3>
            <p><code>GET /</code> — this dashboard</p>
            <p><code>GET /heavy</code> — heavy CPU load (spike test)</p>
            <p><code>GET /work/light</code> — light CPU load (normal-load test)</p>
            <p><code>GET /metrics</code> — raw metrics JSON</p>
            <p><code>GET /health</code> — health check</p>
        </div>
    </body>
    </html>
    """


@app.route("/metrics")
def metrics():
    """Raw metrics JSON. Read-only — doesn't interfere with
    monitoring/collector.py's own reset_request_count() cycle."""
    return jsonify({
        "request_count_since_last_poll": get_request_count(),
        "status": "running",
    })

@app.route("/request-rate", methods=["GET"])
def request_rate():
    """Non-destructive peek at the current counter -- unlike
    monitoring/collector.py's reset_request_count(), this never clears it,
    so polling it from the landing page cannot interfere with the
    collector's own read-and-reset cycle."""
    return jsonify({"request_count": get_request_count()})


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
    (Optional) Accepts 10 metric records (each with a 'timestamp') and returns
    {predicted_cpu, upper_bound, anomaly} from model.predict.predict_load().
    Only works from a full repo checkout with the ML stack installed; the
    Docker image deliberately excludes model/, so it answers 503 there.

    FIX: Import is lazy (inside the function) so Flask starts even if
    TensorFlow or the .keras file are not available in this environment.
    Without this fix the entire app crashed on startup when run inside Docker
    (model/ directory is not copied into the container).
    """
    try:
        from model.predict import predict_load     # lazy import — safe
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

    predicted_cpu, upper_bound, anomaly = predict_load(data["records"])
    return jsonify({
        "predicted_cpu": predicted_cpu,
        "upper_bound": upper_bound,
        "anomaly": anomaly,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)