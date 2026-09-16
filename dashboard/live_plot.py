"""
dashboard/live_plot.py
Person D — Streamlit live monitoring dashboard.

Run:
    streamlit run dashboard/live_plot.py
    streamlit run dashboard/live_plot.py -- --demo

Live mode reads logs/events.csv (written by main.py's log_event).
Demo mode generates synthetic data so the UI can be built/tested
before the full pipeline is running.
"""
import os
import time
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "events.csv"
)
REFRESH_SECONDS = 5
THRESHOLD = 70.0


# ── Data loading ──────────────────────────────────────────────────────────────
def load_live() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
        return df
    except Exception as e:
        st.warning(f"Could not read log: {e}")
        return pd.DataFrame()


def generate_demo(n: int = 40) -> pd.DataFrame:
    """Synthetic data resembling a spike + proactive scale."""
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
        rows.append({
            "timestamp": t,
            "actual_cpu": cpu,
            "predicted_cpu": pred,
            "upper_bound": upper,
            "anomaly_flag": anomaly,
            "signal": signal,
            "replicas": 1 + (1 if i >= int(n * 0.55) else 0)
                        - (1 if i >= int(n * 0.85) else 0),
            "shap_cpu": shap_cpu,
            "shap_memory": shap_mem,
            "shap_request": shap_req,
        })
    return pd.DataFrame(rows)


# ── Charts ────────────────────────────────────────────────────────────────────
def cpu_chart(df: pd.DataFrame):
    fig = go.Figure()

    # Confidence band
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

    # Threshold line
    fig.add_hline(y=THRESHOLD, line_dash="dot",
                  line_color="red",
                  annotation_text=f"Threshold {THRESHOLD}%",
                  annotation_position="top left")

    # Anomaly markers
    anom = df[df["anomaly_flag"] == True]
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


def lead_time(df: pd.DataFrame) -> float:
    """Seconds between last scale_up and the next threshold breach."""
    ups = df[df["signal"] == "scale_up"]
    if ups.empty:
        return 0.0
    t_scale = ups.iloc[-1]["timestamp"]
    after = df[(df["timestamp"] > t_scale) &
               (df["actual_cpu"] >= THRESHOLD)]
    if after.empty:
        return 0.0
    return (after.iloc[0]["timestamp"] - t_scale).total_seconds()


# ── Render ────────────────────────────────────────────────────────────────────
def render(df: pd.DataFrame):
    st.set_page_config(page_title="ProximaScale", layout="wide")
    st.title("ProximaScale — Live Auto-Scaling Dashboard")

    if df.empty:
        st.info("No data yet. Run `python main.py` or use `--demo`.")
        return

    df = df.tail(30).reset_index(drop=True)

    # ── Top metric cards ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current replicas", int(df.iloc[-1]["replicas"]))
    c2.metric("Actual CPU%", f"{df.iloc[-1]['actual_cpu']:.1f}")
    c3.metric("Predicted CPU%", f"{df.iloc[-1]['predicted_cpu']:.1f}")
    c4.metric("Lead time (s)", f"{lead_time(df):.0f}")

    # ── Main chart ────────────────────────────────────────────────────────
    st.subheader("CPU — actual vs predicted (with confidence band)")
    st.plotly_chart(cpu_chart(df), use_container_width=True)

    # ── SHAP + info ───────────────────────────────────────────────────────
    left, right = st.columns([1, 1])
    with left:
        st.subheader("SHAP — why did we scale?")
        last_up = df[df["signal"] == "scale_up"]
        if not last_up.empty:
            row = last_up.iloc[-1]
            pct = {
                "CPU%": float(row["shap_cpu"]),
                "Memory%": float(row["shap_memory"]),
                "Request Rate": float(row["shap_request"]),
            }
            if sum(pct.values()) > 0:
                st.plotly_chart(shap_chart(pct), use_container_width=True)
                st.caption(
                    f"Last scale_up driven by: "
                    + ", ".join(f"{k} {v:.0f}%" for k, v in pct.items())
                )
            else:
                st.info("No SHAP values recorded for the last scale_up.")
        else:
            st.info("No scale_up event yet.")

    with right:
        st.subheader("Recent events")
        ev = df[df["signal"] != "hold"][
            ["timestamp", "signal", "actual_cpu", "predicted_cpu", "replicas"]
        ]
        st.dataframe(ev.tail(10), use_container_width=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args, _ = parser.parse_known_args()

    if args.demo:
        df = generate_demo()
        render(df)
        time.sleep(REFRESH_SECONDS)
        st.rerun()
    else:
        df = load_live()
        render(df)
        time.sleep(REFRESH_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()