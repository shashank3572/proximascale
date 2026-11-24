"""
dashboard/live_plot.py
Person D -- Streamlit live monitoring dashboard.

Run:
    streamlit run dashboard/live_plot.py
    streamlit run dashboard/live_plot.py -- --demo

Live mode reads logs/events.csv (written by main.py's log_event).
The collector appends live metrics to data/collected/metrics_live.csv --
data/collected/metrics.csv stays the frozen 90k synthetic training dataset.

Consistency rule (audit M2)
---------------------------
The chart draws the EFFECTIVE thresholds the engine logged for each decision
(upper_threshold/lower_threshold columns), not a value re-derived from a config
file. A config value can disagree with what the engine used: the adaptive band
is clamped and window-dependent, and an operator may have edited config.yaml
between runs. The config value is only a fallback for logs written before the
threshold columns existed.
"""
import os
import time
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT_DIR, "logs", "events.csv")
DEFAULT_CONFIG_PATH = os.environ.get(
    "PROXIMASCALE_CONFIG_PATH",
    os.path.join(ROOT_DIR, "config.yaml"),
)
REFRESH_SECONDS = 5
FALLBACK_THRESHOLD = 75.0  # only used if the config file can't be read at all

NUMERIC_COLUMNS = [
    "actual_cpu", "predicted_cpu", "upper_bound", "upper_threshold",
    "lower_threshold", "replicas", "request_rate", "shap_cpu", "shap_memory",
    "shap_request",
]


def load_threshold(config_path: str) -> float:
    """cpu_upper_threshold from whichever config the controller ran with.

    Fallback only: prefer the per-row `upper_threshold` the engine logged.
    """
    try:
        with open(config_path, "r") as f:
            return float(yaml.safe_load(f).get("cpu_upper_threshold", FALLBACK_THRESHOLD))
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        return FALLBACK_THRESHOLD


# ── Data loading ──────────────────────────────────────────────────────────────
def load_live() -> pd.DataFrame:
    """Read logs/events.csv, tolerating a missing/partial/malformed file."""
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(LOG_PATH)
    except Exception as e:
        st.warning(f"Could not read log: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # A crash mid-write can leave a truncated last line; parse_dates on the
    # whole column would raise and blank the dashboard.
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    except (KeyError, ValueError, TypeError):
        return pd.DataFrame()
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ("signal", "reason", "forecast_mode"):
        if column not in df.columns:
            df[column] = ""
    df["signal"] = df["signal"].fillna("").astype(str)
    df["anomaly_flag"] = df.get("anomaly_flag", False)
    return df.reset_index(drop=True)


def generate_demo(n: int = 40) -> pd.DataFrame:
    """Synthetic data resembling a spike + proactive scale (demo only)."""
    now = datetime.now()
    rows = []
    for i in range(n):
        t = now - timedelta(seconds=(n - i) * 10)
        base = 30 + i * 1.5
        spike = 25 if i > n * 0.6 else 0
        cpu = min(95, base + spike + np.random.normal(0, 2))
        pred = cpu + (10 if i > n * 0.55 else -5)
        upper = pred + 6
        anomaly = bool(i in (n - 4, n - 3))
        signal = "hold"
        if i == int(n * 0.55):
            signal = "scale_up"
        if i == int(n * 0.85):
            signal = "scale_down"
        shap_cpu, shap_mem, shap_req = 0.0, 0.0, 0.0
        if signal == "scale_up":
            shap_cpu, shap_mem, shap_req = 45.0, 32.0, 23.0
        reason = "in_band"
        if signal == "scale_up":
            reason = "upper_bound_risk" if anomaly is False else "anomaly"
        elif signal == "scale_down":
            reason = "cpu_low"
        rows.append({
            "timestamp": t,
            "actual_cpu": cpu,
            "predicted_cpu": pred,
            "upper_bound": upper,
            "upper_threshold": 75.0 + (5.0 if i > n * 0.5 else 0.0),
            "lower_threshold": 30.0,
            "anomaly_flag": anomaly,
            "signal": signal,
            "replicas": 1 + (1 if i >= int(n * 0.55) else 0)
                        - (1 if i >= int(n * 0.85) else 0),
            "request_rate": int(0 if i < n * 0.6 else 40 + i),
            "forecast_mode": "demo",
            "shap_cpu": shap_cpu,
            "shap_memory": shap_mem,
            "shap_request": shap_req,
            "reason": reason,
        })
    return pd.DataFrame(rows)


# ── Charts ────────────────────────────────────────────────────────────────────
def threshold_series(df: pd.DataFrame, fallback: float = FALLBACK_THRESHOLD):
    """(upper, lower) series actually used, per row.

    Prefers the engine's logged effective thresholds. Falls back to the
    static config value (upper) / static-minus-45 (lower) for logs written
    before schema v2, so an old file still renders.
    """
    if "upper_threshold" in df.columns and df["upper_threshold"].notna().any():
        upper = df["upper_threshold"].ffill().bfill().fillna(fallback)
    else:
        upper = pd.Series([fallback] * len(df), index=df.index, dtype=float)

    if "lower_threshold" in df.columns and df["lower_threshold"].notna().any():
        lower = df["lower_threshold"].ffill().bfill()
    else:
        lower = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    return upper, lower


def cpu_chart(df: pd.DataFrame, threshold: float = FALLBACK_THRESHOLD):
    fig = go.Figure()
    upper, lower = threshold_series(df, threshold)

    # Confidence band (predicted -> upper bound)
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["upper_bound"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["predicted_cpu"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(255,165,0,0.2)",
        name="Confidence band", hoverinfo="skip",
    ))

    # Actual / predicted
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["actual_cpu"],
        mode="lines", name="Actual CPU%",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["predicted_cpu"],
        mode="lines", name="Predicted CPU%",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))

    # Effective adaptive band (what the engine actually compared against)
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=upper,
        mode="lines", name="Upper threshold (effective)",
        line=dict(color="red", width=1.5, dash="dot"),
    ))
    if lower.notna().any():
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=lower,
            mode="lines", name="Lower threshold (effective)",
            line=dict(color="#2ca02c", width=1.5, dash="dot"),
        ))

    # Anomaly markers
    if "anomaly_flag" in df.columns:
        anom = df[df["anomaly_flag"].astype(str).str.lower().isin(["true", "1"])]
        if not anom.empty:
            fig.add_trace(go.Scatter(
                x=anom["timestamp"], y=anom["actual_cpu"],
                mode="markers", name="Anomaly",
                marker=dict(color="red", size=12, symbol="x"),
            ))

    # Scale markers
    ups = df[df["signal"] == "scale_up"]
    downs = df[df["signal"] == "scale_down"]
    if not ups.empty:
        fig.add_trace(go.Scatter(
            x=ups["timestamp"], y=ups["actual_cpu"],
            mode="markers", name="scale_up",
            marker=dict(color="green", size=16, symbol="triangle-up"),
        ))
    if not downs.empty:
        fig.add_trace(go.Scatter(
            x=downs["timestamp"], y=downs["actual_cpu"],
            mode="markers", name="scale_down",
            marker=dict(color="purple", size=16, symbol="triangle-down"),
        ))

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="CPU %", xaxis_title="",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def traffic_chart(df: pd.DataFrame):
    """Requests per collector poll. Unit is requests per window, NOT per second
    -- the label says so, because mixing those two definitions was audit C4."""
    if "request_rate" not in df.columns or df["request_rate"].isna().all():
        return None
    fig = go.Figure(go.Bar(
        x=df["timestamp"], y=df["request_rate"],
        marker_color="#9467bd", name="Requests",
    ))
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="requests / poll window", xaxis_title="",
    )
    return fig


def shap_chart(pct: dict):
    fig = go.Figure(go.Bar(
        x=list(pct.values()),
        y=list(pct.keys()),
        orientation="h",
        marker_color=["#1f77b4", "#2ca02c", "#ff7f0e"],
    ))
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Contribution %", yaxis=dict(autorange="reversed"),
    )
    return fig


def lead_time(df: pd.DataFrame, threshold: float = FALLBACK_THRESHOLD) -> float:
    """Seconds between a scale_up and the threshold breach it preempted.

    A proactive scale-up fires BEFORE the breach, so the primary reading is
    the next breach after the newest scale_up. But a healthy proactive
    response can hold the container below the threshold for the whole visible
    window -- which used to display as lead time 0, indistinguishable from
    having scaled too late. Fallback: measure from the newest scale_up
    BACKWARD to the most recent breach (how long ago we reacted to the last
    one), which stays positive while the fleet holds steady.
    """
    ups = df[df["signal"] == "scale_up"]
    if ups.empty:
        return 0.0
    t_scale = ups.iloc[-1]["timestamp"]

    after = df[(df["timestamp"] > t_scale) &
               (df["actual_cpu"] >= threshold)]
    if not after.empty:
        return (after.iloc[0]["timestamp"] - t_scale).total_seconds()

    before = df[(df["timestamp"] < t_scale) &
                (df["actual_cpu"] >= threshold)]
    if before.empty:
        return 0.0
    return (t_scale - before.iloc[-1]["timestamp"]).total_seconds()


# ── Render ────────────────────────────────────────────────────────────────────
def _staleness(df: pd.DataFrame):
    try:
        age = (datetime.now() - df.iloc[-1]["timestamp"].to_pydatetime()).total_seconds()
    except Exception:
        return None
    return age


def render(df: pd.DataFrame, threshold: float = FALLBACK_THRESHOLD, is_live: bool = False):
    st.set_page_config(page_title="ProximaScale", layout="wide")
    st.title("ProximaScale — Live Auto-Scaling Dashboard")

    if df is None or df.empty:
        st.info("No data yet. Run `python main.py` or use `--demo`.")
        return

    df = df.tail(60).reset_index(drop=True)
    for column in ("actual_cpu", "predicted_cpu", "upper_bound", "replicas"):
        if column not in df.columns:
            df[column] = np.nan

    upper, _lower = threshold_series(df, threshold)
    last = df.iloc[-1]

    # ── Data freshness (never present stale data as live) ─────────────────
    if is_live:
        age = _staleness(df)
        if age is None:
            st.warning("Timestamps unreadable — freshness unknown.")
        elif age > 90:
            st.warning(
                f"Data is stale: newest decision was {age:.0f}s ago. "
                "Is `python main.py` still running?"
            )
        else:
            st.caption(f"Live — newest decision {age:.0f}s ago. "
                       f"Forecast mode: {last.get('forecast_mode', 'unknown')}")

    # ── Top metric cards ──────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    reps = last.get("replicas")
    c1.metric("Replicas (running workers)",
              "?" if pd.isna(reps) else int(reps))
    c2.metric("Actual CPU%", "?" if pd.isna(last["actual_cpu"]) else f"{last['actual_cpu']:.1f}")
    c3.metric("Predicted CPU%",
              "?" if pd.isna(last["predicted_cpu"]) else f"{last['predicted_cpu']:.1f}")
    c4.metric("Upper threshold (effective)", f"{float(upper.iloc[-1]):.1f}")
    c5.metric("Lead time (s)", f"{lead_time(df, float(upper.iloc[-1])):.0f}")

    # ── Main chart ────────────────────────────────────────────────────────
    st.subheader("CPU — actual vs predicted, with the effective adaptive band")
    st.plotly_chart(cpu_chart(df, threshold), use_container_width=True)

    traffic = traffic_chart(df)
    if traffic is not None:
        st.subheader("Workload traffic")
        st.plotly_chart(traffic, use_container_width=True)
        st.caption(
            "`request_rate` counts workload requests inside ONE collector poll "
            "window (default 30 s) — not requests per second. The app also "
            "reports the measured window length at /request-rate/reset."
        )

    # ── SHAP + info ───────────────────────────────────────────────────────
    left, right = st.columns([1, 1])
    with left:
        st.subheader("SHAP — why did we scale?")
        last_up = df[df["signal"] == "scale_up"]
        if not last_up.empty:
            row = last_up.iloc[-1]
            pct = {
                "CPU%": float(row.get("shap_cpu") or 0.0),
                "Memory%": float(row.get("shap_memory") or 0.0),
                "Request Rate": float(row.get("shap_request") or 0.0),
            }
            if sum(pct.values()) > 0:
                st.plotly_chart(shap_chart(pct), use_container_width=True)
                st.caption(
                    "Last scale_up driven by: "
                    + ", ".join(f"{k} {v:.0f}%" for k, v in pct.items())
                )
            elif is_live:
                st.info(
                    "No SHAP attribution recorded for the last scale_up. "
                    "main.py computes it on every scale-up; if you just "
                    "upgraded, the newest scale_up may predate it."
                )
            else:
                st.info("No SHAP values recorded for the last scale_up.")
        else:
            st.info("No scale_up event yet.")

    with right:
        st.subheader("Recent events")
        cols = ["timestamp", "signal", "reason", "actual_cpu", "predicted_cpu",
                "upper_threshold", "replicas", "forecast_mode"]
        cols = [c for c in cols if c in df.columns]     # tolerate older logs
        ev = df[df["signal"] != "hold"][cols] if "signal" in df.columns else df[cols]
        st.dataframe(ev.tail(10), use_container_width=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH,
        help="Path to the same scaling-rules config main.py was run with "
             "(only used as a fallback for logs without logged thresholds).",
    )
    args, _ = parser.parse_known_args()
    threshold = load_threshold(args.config)

    try:
        df = generate_demo() if args.demo else load_live()
        render(df, threshold, is_live=not args.demo)
    except Exception as e:
        # A dashboard must never crash the operator out of the page while the
        # rest of the system is still running.
        st.error(f"Dashboard error while rendering: {e}")
        return

    time.sleep(REFRESH_SECONDS)
    st.rerun()


if __name__ == "__main__":
    main()
