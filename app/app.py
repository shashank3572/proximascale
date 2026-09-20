"""
ProximaScale Flask Application
------------------------------

Flask workload/API service for ProximaScale.

Routes:
    GET  /              → landing/status page
    GET  /work/light    → light CPU workload
    GET  /work/heavy    → heavy CPU workload
    GET  /metrics       → current request metrics
    GET  /request-rate  → current request count for monitoring window
    GET  /health        → health check
    POST /predict       → optional direct ML prediction API

The Streamlit dashboard remains the main visualization interface.
This Flask page is intentionally lightweight and is primarily used
for service status, workload testing, and API demonstration.
"""

from flask import Flask, request, jsonify
import math
import sys
import os

# Project-root imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from monitoring.metrics import increment_request_count, get_request_count


app = Flask(__name__)


# Routes that should NOT contribute to workload request metrics.
# Otherwise refreshing the Flask page would affect the monitoring data.
_EXCLUDED_FROM_COUNT = {
    "/",
    "/health",
    "/request-rate",
    "/metrics",
    "/predict",
    "/favicon.ico",
}


@app.before_request
def count_request():
    """Count only actual workload/API traffic."""
    if request.path not in _EXCLUDED_FROM_COUNT:
        increment_request_count()


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

@app.route("/")
def landing_page():
    """Sem-1-style Flask landing page, enhanced for ProximaScale."""

    count = get_request_count()

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>ProximaScale</title>

    <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 0;
            background: #f4f7fb;
            color: #263238;
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                Roboto,
                Arial,
                sans-serif;
        }}

        .container {{
            max-width: 820px;
            margin: 55px auto;
            padding: 0 20px;
        }}

        .header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 8px;
        }}

        h1 {{
            margin: 0;
            font-size: 30px;
            color: #263238;
        }}

        .subtitle {{
            margin-top: 7px;
            color: #667085;
            font-size: 14px;
        }}

        .status {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 7px 12px;
            border-radius: 20px;
            background: #e8f7ee;
            color: #18794e;
            font-size: 13px;
            font-weight: 600;
            white-space: nowrap;
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #20a464;
        }}

        .divider {{
            height: 2px;
            background: #69aeb3;
            margin: 12px 0 20px;
        }}

        .card {{
            background: white;
            border-radius: 10px;
            padding: 23px 25px;
            margin-bottom: 17px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.07);
            border: 1px solid #edf0f4;
        }}

        .card-title {{
            display: flex;
            align-items: center;
            gap: 9px;
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 13px;
        }}

        .request-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
        }}

        .request-count {{
            font-size: 35px;
            font-weight: 750;
            color: #2794c2;
            line-height: 1.1;
        }}

        .request-label {{
            margin-top: 5px;
            color: #7b8794;
            font-size: 12px;
        }}

        button {{
            border: none;
            border-radius: 6px;
            padding: 10px 15px;
            background: #319bc9;
            color: white;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
        }}

        button:hover {{
            background: #2587b2;
        }}

        .refresh-info {{
            margin-top: 12px;
            color: #8a94a3;
            font-size: 12px;
        }}

        .endpoint {{
            display: flex;
            align-items: center;
            gap: 9px;
            margin: 11px 0;
            font-size: 14px;
        }}

        .method {{
            min-width: 42px;
            padding: 4px 7px;
            border-radius: 4px;
            background: #eef5f8;
            color: #32738d;
            font-family: monospace;
            font-size: 11px;
            font-weight: 700;
            text-align: center;
        }}

        code {{
            color: #394b59;
            font-family: Consolas, monospace;
        }}

        .endpoint a {{
            color: #34495e;
            text-decoration: none;
        }}

        .endpoint a:hover {{
            color: #1884b4;
            text-decoration: underline;
        }}

        .description {{
            color: #737f8c;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }}

        .info {{
            padding: 13px;
            border-radius: 7px;
            background: #f7f9fb;
            border: 1px solid #edf0f3;
        }}

        .info-title {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #8994a0;
            margin-bottom: 5px;
        }}

        .info-value {{
            font-size: 13px;
            font-weight: 650;
            color: #3d4d5a;
        }}

        footer {{
            text-align: center;
            margin-top: 25px;
            color: #9aa3ad;
            font-size: 11px;
        }}

        @media (max-width: 650px) {{
            .header {{
                align-items: flex-start;
                flex-direction: column;
            }}

            .request-row {{
                align-items: flex-start;
                flex-direction: column;
            }}

            .info-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>

<body>

<div class="container">

    <div class="header">
        <div>
            <h1>ProximaScale Flask App</h1>
            <div class="subtitle">
                Proactive autoscaling workload & API service
            </div>
        </div>

        <div class="status">
            <span class="status-dot"></span>
            RUNNING
        </div>
    </div>

    <div class="divider"></div>


    <!-- Request counter -->
    <div class="card">

        <div class="card-title">
            📈 Current Request Activity
        </div>

        <div class="request-row">

            <div>
                <div id="request-count" class="request-count">
                    {count}
                </div>

                <div class="request-label">
                    requests since last monitoring reset
                </div>
            </div>

            <button onclick="refreshRequestCount()">
                ↻ Refresh
            </button>

        </div>

        <div class="refresh-info">
            Automatically refreshed every 3 seconds.
            This page itself does not contribute to the request count.
        </div>

    </div>


    <!-- Workload endpoints -->
    <div class="card">

        <div class="card-title">
            ⚡ Workload Endpoints
        </div>

        <div class="endpoint">
            <span class="method">GET</span>
            <a href="/work/light" target="_blank">
                <code>/work/light</code>
            </a>
            <span class="description">
                — normal/light CPU workload
            </span>
        </div>

        <div class="endpoint">
            <span class="method">GET</span>
            <a href="/work/heavy" target="_blank">
                <code>/work/heavy</code>
            </a>
            <span class="description">
                — heavy/spike CPU workload
            </span>
        </div>

    </div>


    <!-- API endpoints -->
    <div class="card">

        <div class="card-title">
            🔗 Available API Endpoints
        </div>

        <div class="endpoint">
            <span class="method">GET</span>
            <a href="/health" target="_blank">
                <code>/health</code>
            </a>
            <span class="description">
                — service health check
            </span>
        </div>

        <div class="endpoint">
            <span class="method">GET</span>
            <a href="/metrics" target="_blank">
                <code>/metrics</code>
            </a>
            <span class="description">
                — current monitoring metrics
            </span>
        </div>

        <div class="endpoint">
            <span class="method">GET</span>
            <a href="/request-rate" target="_blank">
                <code>/request-rate</code>
            </a>
            <span class="description">
                — current request counter
            </span>
        </div>

        <div class="endpoint">
            <span class="method">POST</span>
            <span>
                <code>/predict</code>
            </span>
            <span class="description">
                — ML prediction API
            </span>
        </div>

    </div>


    <!-- Project information -->
    <div class="card">

        <div class="card-title">
            ⚙ ProximaScale Services
        </div>

        <div class="info-grid">

            <div class="info">
                <div class="info-title">Monitoring</div>
                <div class="info-value">SQLite-backed</div>
            </div>

            <div class="info">
                <div class="info-title">Prediction</div>
                <div class="info-value">Residual Stacking</div>
            </div>

            <div class="info">
                <div class="info-title">Scaling</div>
                <div class="info-value">Adaptive Threshold</div>
            </div>

        </div>

    </div>


    <footer>
        ProximaScale · Flask workload service ·
        Streamlit dashboard provides detailed visualization
    </footer>

</div>


<script>

async function refreshRequestCount() {{
    try {{
        const response = await fetch("/request-rate", {{
            cache: "no-store"
        }});

        if (!response.ok) {{
            throw new Error("Request failed");
        }}

        const data = await response.json();

        document.getElementById("request-count").textContent =
            data.request_count;

    }} catch (error) {{
        console.error("Unable to refresh request count:", error);
    }}
}}


// Initial refresh
refreshRequestCount();

// Keep the number live without requiring manual browser refresh
setInterval(refreshRequestCount, 3000);

</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@app.route("/metrics")
def metrics():
    """Read-only monitoring metrics endpoint."""

    return jsonify({
        "request_count_since_last_poll": get_request_count(),
        "status": "running",
    })


@app.route("/request-rate", methods=["GET"])
def request_rate():
    """
    Read-only request counter.

    This endpoint NEVER resets the counter.
    The monitoring collector owns the reset cycle.
    """

    return jsonify({
        "request_count": get_request_count()
    })


# ---------------------------------------------------------------------------
# Workload endpoints
# ---------------------------------------------------------------------------

@app.route("/work/light")
def light_work():
    """Generate a light CPU workload."""

    result = 0

    for i in range(200_000):
        result += math.sqrt(i)

    return "Light work completed"


@app.route("/work/heavy")
def heavy_work():
    """Generate a heavy CPU workload used for spike testing."""

    result = 0

    for i in range(5_000_000):
        result += math.sqrt(i)

    return "Heavy work completed"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    })


# ---------------------------------------------------------------------------
# Optional prediction API
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def get_prediction():
    """
    Optional direct ML prediction endpoint.

    Expects:
        {
            "records": [...]
        }

    Exactly 10 records are required.
    """

    try:
        from model.predict import predict_load
    except ImportError as e:
        return jsonify({
            "error": "ML model not available in this environment.",
            "details": str(e),
        }), 503

    data = request.get_json(silent=True)

    if not data or "records" not in data:
        return jsonify({
            "error": "JSON body must contain 'records'."
        }), 400

    if len(data["records"]) != 10:
        return jsonify({
            "error": "Exactly 10 records required"
        }), 400

    try:
        predicted_cpu, upper_bound, anomaly = predict_load(
            data["records"]
        )

        return jsonify({
            "predicted_cpu": predicted_cpu,
            "upper_bound": upper_bound,
            "anomaly": anomaly,
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed.",
            "details": str(e),
        }), 500


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )