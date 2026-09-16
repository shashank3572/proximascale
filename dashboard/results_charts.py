"""
dashboard/results_charts.py
Person D — Static results figures for the final report.

Reads experiment CSVs from results/ and produces PNGs in
dashboard/results/. Run:

    python dashboard/results_charts.py

Inputs (create these as experiments are run):
    results/exp1_accuracy.csv     -> model, rmse, mae
    results/exp2_lead_time.csv    -> run, lead_time_seconds
    results/exp3_latency.csv      -> condition, avg_ms, p95_ms, replicas
    results/exp4_oscillation.csv  -> variant, oscillation_count
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "dashboard", "results")
os.makedirs(OUT_DIR, exist_ok=True)


def _load(name: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"[skip] {name} not found at {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


# ── Experiment 1 — Prediction accuracy ────────────────────────────────────────
def exp1_accuracy():
    df = _load("exp1_accuracy.csv")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(df))
    w = 0.38
    ax.bar([i - w/2 for i in x], df["rmse"], w, label="RMSE", color="#1f77b4")
    ax.bar([i + w/2 for i in x], df["mae"],  w, label="MAE",  color="#ff7f0e")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"])
    ax.set_ylabel("Error")
    ax.set_title("Exp 1 — Prediction accuracy (lower is better)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "exp1_accuracy.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


# ── Experiment 2 — Lead time ──────────────────────────────────────────────────
def exp2_lead_time():
    df = _load("exp2_lead_time.csv")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(df)), df["lead_time_seconds"],
           color="#2ca02c")
    ax.axhline(60, color="red", linestyle="--", label="Target ≥60s")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"run {i+1}" for i in range(len(df))])
    ax.set_ylabel("Lead time (s)")
    ax.set_title("Exp 2 — Proactive scaling lead time")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "exp2_lead_time.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


# ── Experiment 3 — Response latency ───────────────────────────────────────────
def exp3_latency():
    df = _load("exp3_latency.csv")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(df))
    w = 0.38
    ax.bar([i - w/2 for i in x], df["avg_ms"], w,
           label="Avg latency", color="#1f77b4")
    ax.bar([i + w/2 for i in x], df["p95_ms"], w,
           label="P95 latency", color="#d62728")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["condition"])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Exp 3 — Response latency under spike")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "exp3_latency.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


# ── Experiment 4 — Oscillation ────────────────────────────────────────────────
def exp4_oscillation():
    df = _load("exp4_oscillation.csv")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["variant"], df["oscillation_count"],
           color=["#d62728", "#2ca02c"])
    ax.set_ylabel("Oscillations in 5-min window")
    ax.set_title("Exp 4 — Counterfactual correction effect")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "exp4_oscillation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    exp1_accuracy()
    exp2_lead_time()
    exp3_latency()
    exp4_oscillation()
    print("Done. Figures written to dashboard/results/")