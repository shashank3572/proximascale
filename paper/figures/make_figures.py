"""Generate every figure used in the revised ProximaScale manuscript.

Quantitative figures read committed result CSVs or the validation run in
results/validation_2026-11-25/. The architecture figures are schematics drawn
from the module structure documented in the repository.

Run from the repository root:
    python paper/figures/make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "forecast_metrics"
VALID = ROOT / "results" / "validation_2026-11-25"

BLUE = "#1f4e79"
LIGHT = "#dbe5f1"
GREY = "#7f7f7f"
ORANGE = "#c55a11"
GREEN = "#2e7d32"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "savefig.dpi": 300,
    "figure.dpi": 120,
})


def box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, xy_from, xy_to, text="", rad=0.0, fs=6.5, color=BLUE):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2 + 0.010,
                text, ha="center", va="bottom", fontsize=fs, color=color)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# Fig. 1 - deployed architecture
# --------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0.02, 0.78, 0.22, 0.15, "Workload generator\n(Locust, Flask /work/*)", fs=6.6)
    box(ax, 0.34, 0.78, 0.24, 0.15, "Application container\nproximascale-app :5000", fs=6.6)
    box(ax, 0.68, 0.78, 0.22, 0.15, "Worker pool\nproximascale-worker-*", fs=6.6)
    box(ax, 0.34, 0.53, 0.24, 0.13, "Metric collector\n(Docker stats, 30 s)", fs=6.6)
    box(ax, 0.34, 0.30, 0.24, 0.13, "metrics_live.csv\n(CPU, memory, requests)", fs=6.6)
    box(ax, 0.02, 0.05, 0.19, 0.17, "Preprocessing\n30-reading window\nCF correction, scaling", fs=6.0)
    box(ax, 0.255, 0.05, 0.19, 0.17, "Forecasting\nProphet + 3x LSTM\nresidual, MC dropout", fs=6.0)
    box(ax, 0.49, 0.05, 0.19, 0.17, "Decision engine\nadaptive band,\nhysteresis, cooldown", fs=6.0)
    box(ax, 0.725, 0.05, 0.19, 0.17, "Actuator\nDocker SDK\nscale up / down", fs=6.0)

    arrow(ax, (0.24, 0.855), (0.34, 0.855), "requests")
    arrow(ax, (0.46, 0.78), (0.46, 0.66), "poll")
    arrow(ax, (0.46, 0.53), (0.46, 0.43), "append")
    arrow(ax, (0.34, 0.33), (0.115, 0.22), "read last 30", rad=0.25)
    arrow(ax, (0.21, 0.135), (0.255, 0.135))
    arrow(ax, (0.445, 0.135), (0.49, 0.135), "CPU, U")
    arrow(ax, (0.68, 0.135), (0.725, 0.135), "signal")
    ax.text(0.930, 0.52, "capacity feedback", rotation=90, ha="left", va="center",
            fontsize=6.0, color=ORANGE)
    arrow(ax, (0.82, 0.22), (0.79, 0.78), "", rad=-0.12, color=ORANGE)
    ax.text(0.845, 0.45, "scaling\naction", fontsize=6.0, color=ORANGE)
    arrow(ax, (0.62, 0.05), (0.62, 0.020), "", color=GREEN)
    ax.text(0.635, 0.020, "logs/events.csv -> dashboard", fontsize=6.0,
            color=GREEN, va="bottom")

    ax.text(0.5, 0.985, "ProximaScale deployed architecture (closed control loop)",
            ha="center", va="top", fontsize=9, fontweight="bold")
    save(fig, "fig1_architecture.png")


# --------------------------------------------------------------------------
# Fig. 2 - input window and horizon
# --------------------------------------------------------------------------
def fig_timeline():
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    ax.set_xlim(-31, 4.5); ax.set_ylim(-1.2, 1.6); ax.axis("off")

    ax.scatter(range(-30, 1), [0.35] * 31, s=26, color=(0.72, 0.35, 0.15),
               zorder=2, linewidths=0)
    ax.scatter(range(1, 4), [0.35] * 3, s=34, facecolors="white",
               edgecolors=ORANGE, linewidths=1.0, zorder=2)
    for k in range(1, 4):
        ax.text(k, 0.35, f"{k}", ha="center", va="center", fontsize=6.0, color=ORANGE)

    ax.plot([-30, 0], [0.05, 0.05], color=GREY, lw=1)
    ax.plot([1, 3], [0.05, 0.05], color=ORANGE, lw=1, ls="--")
    ax.annotate("", xy=(-30, 0.05), xytext=(0, 0.05),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8))
    ax.text(-15, -0.30, "input window: 30 readings x 30 s = 15 min",
            ha="center", fontsize=7.5, color=GREY)
    ax.annotate("", xy=(3, 1.02), xytext=(1, 1.02),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=0.8))
    ax.text(2, 1.10, "forecast horizon\n3 x 30 s = 90 s", ha="center", fontsize=7.5,
            color=ORANGE)
    ax.text(-15, 0.95, "channels: corrected CPU, memory, request count,\n"
                       "Prophet yhat(t), Prophet yhat(t+90 s)",
            ha="center", fontsize=7, color="#333333")
    ax.plot([0, 0], [-0.10, 0.62], color=BLUE, lw=1.0, ls=":")
    ax.text(0, -0.50, "t (decision instant)", ha="center", fontsize=7.5, color=BLUE)
    ax.text(1.6, -0.85, "the decision uses the t+3 (90 s) forecast",
            ha="center", fontsize=7, style="italic")
    save(fig, "fig2_window_horizon.png")


# --------------------------------------------------------------------------
# Fig. 3 - forecasting stack
# --------------------------------------------------------------------------
def fig_forecast():
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0.01, 0.42, 0.19, 0.40,
        "Input window\n30 steps x 5 channels\n\nCPU, memory,\nrequests,\nyhat(t), yhat(t+90 s)", fs=6.4)
    box(ax, 0.245, 0.42, 0.235, 0.40,
        "Residual LSTM\n\nLSTM 64 -> Dropout 0.1\nLSTM 32 -> Dropout 0.1\nDense 3 (residual)", fs=6.6)
    box(ax, 0.52, 0.62, 0.22, 0.20, "Sum\ny = yhat + r", fs=7.0,
        fc="#fff2cc", ec=ORANGE)
    box(ax, 0.52, 0.20, 0.22, 0.18, "Prophet\nyhat(t+1..3)", fs=6.8, fc="#f2f2f2", ec=GREY)
    box(ax, 0.79, 0.62, 0.20, 0.22,
        "MC Dropout\n30 passes pooled\nover 3 seed models\n-> mean, sigma", fs=6.2,
        fc="#e8f5e9", ec=GREEN)
    box(ax, 0.79, 0.20, 0.20, 0.18,
        "Served output\npoint forecast +\ncalibrated 95% bound\n(live CPU anchored)", fs=6.2,
        fc="#fce4d6", ec=ORANGE)

    arrow(ax, (0.20, 0.62), (0.245, 0.62))
    arrow(ax, (0.48, 0.66), (0.52, 0.70))
    arrow(ax, (0.63, 0.38), (0.63, 0.62), "yhat(t+1..3)", fs=6.2)
    arrow(ax, (0.74, 0.72), (0.79, 0.72), "add")
    arrow(ax, (0.89, 0.62), (0.89, 0.38), "bound")

    ax.text(0.5, 0.97, "Served forecasting stack (bundle proximascale_v4)",
            ha="center", va="top", fontsize=9, fontweight="bold")
    ax.text(0.01, 0.06, "30 435 trainable parameters per member; 3 members (seeds 0/1/2); "
            "Prophet fitted once on the training period", fontsize=6.0, color=GREY)
    save(fig, "fig3_forecasting_stack.png")


# --------------------------------------------------------------------------
# Fig. 4 - accuracy comparison
# --------------------------------------------------------------------------
def fig_accuracy():
    base = {r["method"]: r for r in read_csv(RESULTS / "baseline" / "table_ii.csv")}
    order = [("persistence", "Persistence\n(reactive)"),
             ("linear", "Linear\nextrapolation"),
             ("prophet", "Prophet\nalone"),
             ("univariate", "Univariate\nLSTM"),
             ("hybrid", "ProximaScale\n(10-step)")]
    labels = [l for _, l in order]
    rmse = [float(base[m]["rmse"]) for m, _ in order]
    mae = [float(base[m]["mae"]) for m, _ in order]

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    x = np.arange(len(order)); w = 0.38
    b1 = ax.bar(x - w / 2, rmse, w, label="RMSE", color=BLUE)
    b2 = ax.bar(x + w / 2, mae, w, label="MAE", color="#d9a441")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12,
                    f"{b.get_height():.2f}", ha="center", fontsize=6.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Error (CPU percentage points)")
    ax.set_ylim(0, 12)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Forecast error at t+90 s, 17 988-window test split")
    save(fig, "fig4_accuracy.png")


# --------------------------------------------------------------------------
# Fig. 5 - lead-time distribution per spike
# --------------------------------------------------------------------------
def fig_lead():
    spikes = read_csv(RESULTS / "baseline" / "spikes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, method, title, color in (
            (axes[0], "hybrid", "ProximaScale (10-step)", ORANGE),
            (axes[1], "persistence", "Persistence (reactive)", GREY)):
        leads = [float(r["lead_s"]) for r in spikes if r["method"] == method and r["seed"] == "0"]
        counts = [sum(1 for l in leads if l == 30 * k) for k in range(1, 4)]
        missed = sum(1 for l in leads if l == 0)
        bars = ax.bar(["+30 s", "+60 s", "+90 s", "missed"], counts + [missed],
                      color=[color] * 3 + ["#bbbbbb"])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                    f"{int(b.get_height())}", ha="center", fontsize=6.5)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, max(20, max(counts + [missed]) + 4))
        ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    axes[0].set_ylabel("Spikes (of 36)")
    fig.suptitle("Lead time at the 75 % threshold, per generated spike", fontsize=9)
    fig.tight_layout()
    save(fig, "fig5_lead_time.png")


# --------------------------------------------------------------------------
# Fig. 6 - reproduced decision replay
# --------------------------------------------------------------------------
def fig_decisions():
    def load(path):
        return {r["policy"]: r for r in read_csv(path)}

    shipped = load(VALID / "replay_v4" / "v4_verify" / "summary.csv")
    nofloor = load(VALID / "replay_v4_nofloor" / "v4_verify" / "summary.csv")
    policies = [
        ("Reactive", shipped["P0 Reactive"], GREY),
        ("Univariate", shipped["P1 Univariate LSTM"], "#8faadc"),
        ("ProximaScale", shipped["P2 ProximaScale mean"], BLUE),
        ("Shipped config", shipped["P9 Tuned (config.yaml)"], ORANGE),
        ("Config, floor off", nofloor["P9 Tuned (config.yaml)"], GREEN),
    ]
    metrics = [
        ("Spikes anticipated\n(of 36)", "anticipated", 36.0),
        ("False scale-ups\nper day", "false_ups_per_day", None),
        ("Reversals\nwithin 5 min", "reversals_5min", None),
        ("Mean replicas", "mean_replicas", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.2))
    for j, (ax, (title, col, denom)) in enumerate(zip(axes.ravel(), metrics)):
        vals = [float(r[col]) for _, r, _ in policies]
        bars = ax.barh(range(len(policies)), vals, color=[c for _, _, c in policies], height=0.6)
        for bar, v in zip(bars, vals):
            ax.text(v, bar.get_y() + bar.get_height() / 2, f" {v:g}",
                    ha="left", va="center", fontsize=6.2)
        ax.set_title(title, fontsize=7.5)
        ax.set_yticks(range(len(policies)))
        if j % 2 == 0:
            ax.set_yticklabels([l for l, _, _ in policies], fontsize=6.2)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25); ax.set_axisbelow(True)
        if denom:
            ax.set_xlim(0, denom * 1.25)
        else:
            ax.set_xlim(0, max(vals) * 1.35 if max(vals) else 1)
    fig.suptitle("Reproduced decision replay on the v4 test dump (17 988 decisions, 36 spikes)",
                 fontsize=8.2)
    fig.tight_layout()
    save(fig, "fig6_decision_replay.png")


# --------------------------------------------------------------------------
# Fig. 7 - uncertainty calibration
# --------------------------------------------------------------------------
def fig_uncertainty():
    rel = {float(r["k"]): r for r in read_csv(RESULTS / "baseline" / "reliability.csv")}
    ks = sorted(rel)
    nominal = [float(rel[k]["nominal"]) for k in ks]
    empiric = [float(rel[k]["coverage"]) for k in ks]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=0.9, label="ideal")
    ax.plot(nominal, empiric, "o-", color=BLUE, lw=1.2, ms=4,
            label="MC-Dropout, mean + k sigma")
    for k, n_, e_ in zip(ks, nominal, empiric):
        ax.annotate(f"k={k:g}", (n_, e_), textcoords="offset points", xytext=(4, -8),
                    fontsize=5.8, color=BLUE)
    ax.set_xlabel("Nominal one-sided coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.6, 1.02); ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.25); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("Raw MC-Dropout sigma is miscalibrated", fontsize=8.5)
    save(fig, "fig7_uncertainty.png")


# --------------------------------------------------------------------------
# Dataset statistics
# --------------------------------------------------------------------------
def dataset_stats():
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "collected" / "metrics.csv")
    post = df["post_scaling"].astype(str).str.lower() == "true"
    stats = {
        "rows": int(len(df)),
        "start": str(df["timestamp"].iloc[0]),
        "end": str(df["timestamp"].iloc[-1]),
        "post_scaling_rows": int(post.sum()),
        "cpu_min": float(df.cpu_percent.min()),
        "cpu_mean": float(df.cpu_percent.mean()),
        "cpu_max": float(df.cpu_percent.max()),
        "cpu_p95": float(df.cpu_percent.quantile(0.95)),
        "mem_min": float(df.memory_percent.min()),
        "mem_mean": float(df.memory_percent.mean()),
        "mem_max": float(df.memory_percent.max()),
        "req_min": int(df.request_rate.min()),
        "req_mean": float(df.request_rate.mean()),
        "req_max": int(df.request_rate.max()),
    }
    VALID.mkdir(parents=True, exist_ok=True)
    with open(VALID / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[ok] {(VALID / 'dataset_stats.json').relative_to(ROOT)}")
    return stats


if __name__ == "__main__":
    fig_architecture()
    fig_timeline()
    fig_forecast()
    fig_accuracy()
    fig_lead()
    fig_decisions()
    fig_uncertainty()
    dataset_stats()
