"""
ProximaScale - Evaluation on data no model has seen
---------------------------------------------------
Every number so far comes from one generated series (metrics.csv, seed 42):
the models were trained on its first 80%, chosen on the next 10% and tested
on the last 20%. This script scores the same models, unchanged, on data none
of them (nor Prophet) was fit on:

    fresh     model/generate_realistic_data.py with a new seed (7), one week
              dated after metrics.csv ends, so Prophet extrapolates as it
              would live. Same process, new draws.
    shifted   the same generator (seed 11), but spikes ramp over 3 minutes
              instead of 90 s, hold 4 minutes instead of 7, decay over 3, and
              come twice as often: a change in the process the models
              never saw.
    real_dev  data/collected/metrics_real_dev.csv, the only real container
              data in the repo: 121 idle readings ~63 s apart (CPU ~0.03%).
              A sanity check only: far outside the training range, and
              "t+3" there is ~3 minutes ahead, not 90 s.

Models (all scored with variant_eval.py's inference path, 30 MC passes,
3-seed ensembles pooled): persistence, Prophet, the original committed
ProximaScale, the fair ProximaScale, univariate LSTM, multivariate direct
LSTM (C1), v2 and v3 (live). Forecast metrics are forecast_metrics.py's
(RMSE/MAE against recorded CPU at t+3, lead time and false alarms at 75%),
with v3 minus each model's RMSE and a block-bootstrap 95% CI. For v3 the
calibrated 95% bound's coverage is reported, and the decision engine is
replayed (tune_decisions.py's replay) with the tuned config.yaml settings,
the original live config, the forecast mean alone and the reactive signal.

    python model/unseen_eval.py
Writes results/unseen/: data/{fresh,shifted}.csv, forecast.csv,
significance.csv, coverage.csv, decisions.csv and meta.json.
"""

import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import forecast_metrics as F  # noqa: E402
import generate_realistic_data as G  # noqa: E402
import train_variants as T  # noqa: E402
import tune_decisions as TD  # noqa: E402
import variant_eval as V  # noqa: E402
from preprocessing import FEATURE_COLS, load_csv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "unseen"
REAL_CSV = REPO_ROOT / "data" / "collected" / "metrics_real_dev.csv"
V3_BUNDLE = REPO_ROOT / "model" / "saved" / "proximascale_v3"
CONFIG = REPO_ROOT / "config.yaml"
START = datetime(2026, 10, 21, 0, 0, 0)       # metrics.csv ends 2026-10-20 06:00
WEEK = 7 * 24 * 120                           # rows of 30 s
WARMUP = 60                                    # rows before the first scored window

DATASETS = {
    "fresh": {"SEED": 7},
    "shifted": {"SEED": 11, "SPIKE_RAMP": 6, "SPIKE_HOLD": 8, "SPIKE_DECAY": 6,
                "N_SPIKES": WEEK // 250},
    # defined after "shifted" was seen, before any result on it: the check
    # for the reactive floor and the varied-spike model
    "shifted2": {"SEED": 23, "SPIKE_RAMP": 5, "SPIKE_HOLD": 10, "SPIKE_DECAY": 5,
                 "N_SPIKES": WEEK // 333},
}
VARIED_DIR = REPO_ROOT / "model" / "saved" / "variants_varied"
BOTH_DIR = REPO_ROOT / "model" / "saved" / "variants_both"       # metrics.csv + metrics_varied.csv
MODELS = {                                      # name in the tables -> (variant, runs dir)
    "ProximaScale, fair": ("hybrid", T.VARIANTS_DIR),
    "Univariate LSTM": ("univariate", T.VARIANTS_DIR),
    "Multivariate LSTM, direct (C1)": ("multi_direct", T.VARIANTS_DIR),
    "v2": ("hx_w30_prophet", T.VARIANTS_DIR),
    "v3 (live)": ("hx3_drop10", T.VARIANTS_DIR),
    "v3 trained on varied spikes": ("hx3_drop10", VARIED_DIR),
    "v3 trained on both": ("hx3_drop10", BOTH_DIR),
}
REFERENCE = "v3 (live)"
DECISION_MODELS = ("v3 (live)", "v3 trained on varied spikes", "v3 trained on both")


def _log(message):
    print(f"[ProximaScale] {message}", flush=True)


def generate(name, overrides):
    """generate_realistic_data.generate() with its module settings replaced."""
    path = OUT / "data" / f"{name}.csv"
    settings = {"N_ROWS": WEEK, "START_TIME": START, "N_SPIKES": WEEK // 500,
                "OUTPUT_PATH": path, **overrides}
    saved = {k: getattr(G, k) for k in settings}
    for k, v in settings.items():
        setattr(G, k, v)
    G.SPIKE_LEN = G.SPIKE_RAMP + G.SPIKE_HOLD + G.SPIKE_DECAY
    try:
        G.generate()
    finally:
        for k, v in saved.items():
            setattr(G, k, v)
        G.SPIKE_LEN = G.SPIKE_RAMP + G.SPIKE_HOLD + G.SPIKE_DECAY
    return path


def prophet_yhat(name, timestamps):
    """train_variants.prophet_yhat without Prophet's uncertainty sampling
    (1000 trend draws per row, unused: yhat does not depend on them)."""
    from prophet_model import get_prophet_fitted
    model = T.load_prophet(name)
    model.uncertainty_samples = 0
    return np.asarray(get_prophet_fitted(model, timestamps), dtype=float)


def committed_hybrid(frame):
    with open(V.SAVED / "scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open(V.SAVED / "scaler_residual.pkl", "rb") as f:
        residual_scaler = pickle.load(f)
    paper = {**T.DEFAULTS, "inputs": list(FEATURE_COLS)}
    return V.infer(V.SAVED / "proximascale_lstm.h5",
                   V.scaled(frame.inputs(paper), feature_scaler),
                   residual_scaler, frame.prophet(), residual=True)


def ensemble(variant, frame, root=T.VARIANTS_DIR):
    dirs = sorted((root / variant).glob("seed*"))
    n = V.N_PASSES // len(dirs)
    return V.pool([V.run_variant(variant, d, frame, n_passes=n) for d in dirs])


def score(frame, pred):
    """Scorer without generator truth; tolerant of periods with no onsets."""
    pred = np.clip(pred, 0.0, 100.0)
    err = pred - frame.actual
    row = {"rmse": F.rmse(err), "mae": F.mae(err)}
    onsets = F.onsets_above(frame.cpu, V.THRESHOLD)
    row["onsets"] = len(onsets)
    if len(onsets):
        metrics, _ = F.event_metrics(pred, frame.actual, frame.t, onsets, V.THRESHOLD,
                                     V.HORIZON, V.INTERVAL_S)
        row.update({k: metrics[k] for k in ("lead_mean_s", "missed", "false_alarms_per_day",
                                            "precision", "recall")})
    return row


def evaluate(name, df, rng, base_config, start=None, stop=None):
    _log(f"{name}: {len(df)} rows, {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}")
    prophets = sorted({"committed"} | {T.VARIANTS[v]["prophet"] for v, _ in MODELS.values()})
    yhats = {p: prophet_yhat(p, df["timestamp"]) for p in prophets}
    start = WARMUP if start is None else start
    frame = V.Frame(df, start, len(df) if stop is None else stop, yhats)
    preds = {"Persistence (reactive)": frame.full_cpu[frame.t_abs],
             "Prophet alone": frame.prophet()}
    preds["ProximaScale, original committed"] = committed_hybrid(frame)["mc"]
    outs = {}
    for label, (variant, root) in MODELS.items():
        if not (root / variant).is_dir():
            continue
        _log(f"  {label}")
        outs[label] = ensemble(variant, frame, root)
        preds[label] = outs[label]["mc"]

    rows = [{"dataset": name, "model": m, "windows": len(frame.t), **score(frame, p)}
            for m, p in preds.items()]
    sig = [{"dataset": name, "reference": REFERENCE, "versus": m,
            **{k: v for k, v in V.paired(preds[REFERENCE], p, frame.actual, rng).items()
               if k.startswith("d_rmse")}}
           for m, p in preds.items() if m != REFERENCE and len(frame.t) > 4 * F.BLOCK_LEN]

    # v3's bound: calibrated (bundle config.json, fit on metrics.csv validation) and raw
    cal = json.loads((V3_BUNDLE / "config.json").read_text(encoding="utf-8"))["upper_bound"]
    v3 = outs[REFERENCE]
    cov = []
    for kind, ub in (("calibrated 95%", TD.apply_bound(cal["method"], cal, v3["mc"], v3["sigma"])),
                     ("raw mean + 2 sigma", v3["mc"] + 2 * v3["sigma"])):
        spike = frame.actual > 75
        cov.append({"dataset": name, "bound": kind, "coverage": float(np.mean(frame.actual <= ub)),
                    "coverage_above_75": (float(np.mean(frame.actual[spike] <= ub[spike]))
                                          if spike.any() else np.nan),
                    "mean_width": float(np.mean(ub - v3["mc"]))})

    # decisions: the engine replayed on each v3's forecast mean
    min_cpu = float(base_config.get("anomaly_min_cpu_percent", 5.0))
    dec = []

    def replay(model, label, per, bound, guard, anomaly, floor):
        m = TD.run(per, base_config, bound, guard, anomaly, min_cpu, floor)
        dec.append({"dataset": name, "model": model, "setting": label, **m})

    for model in DECISION_MODELS:
        if model not in outs:
            continue
        o = outs[model]
        per = TD.period(df, frame.t_abs, start, [{"seed": 0, "mean": o["mc"], "sigma": o["sigma"]}])
        replay(model, "tuned config.yaml (guard 75)", per, None, 75.0, None, False)
        replay(model, "tuned + reactive floor", per, None, 75.0, None, True)
        replay(model, "original live config (raw bound + anomaly 2.5)", per, ("raw", None),
               None, 2.5, False)
    now = frame.full_cpu[frame.t_abs]
    reactive = TD.period(df, frame.t_abs, start,
                         [{"seed": 0, "mean": now, "sigma": np.zeros(len(now))}])
    replay("-", "reactive (current CPU)", reactive, None, None, None, False)
    return rows, sig, cov, dec


def main():
    started = time.time()
    import tensorflow as tf
    tf.keras.utils.disable_interactive_logging()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CONFIG, encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}
    rng = np.random.default_rng(F.BOOT_SEED)

    original = REPO_ROOT / "data" / "collected" / "metrics.csv"
    varied = REPO_ROOT / "data" / "collected" / "metrics_varied.csv"
    # name -> (csv, first scored row, end row); validation rows 64800-71999, test 72000-
    sources = {"original_val": (original, 64800, 72000), "original_test": (original, 72000, None)}
    if varied.exists():
        sources["varied_val"] = (varied, 64800, 72000)
        sources["varied_test"] = (varied, 72000, None)
    for name, o in DATASETS.items():
        sources[name] = (generate(name, o), WARMUP, None)
    sources["real_dev"] = (REAL_CSV, 30, None)
    tables = {"forecast": [], "significance": [], "coverage": [], "decisions": []}
    for name, (path, start, stop) in sources.items():
        df = load_csv(path)
        if "post_scaling" not in df.columns:
            df["post_scaling"] = False
        df["post_scaling"] = df["post_scaling"].astype(bool)
        if start > 100:            # only the period plus 100 rows of history (Prophet memory)
            df = df.iloc[start - 100:stop].reset_index(drop=True)
            start, stop = 100, None
        if name == "real_dev":
            gaps = pd.to_datetime(df["timestamp"]).diff().dt.total_seconds()
            _log(f"real_dev: median gap {gaps.median():.0f} s (not 30 s); CPU max "
                 f"{df['cpu_percent'].max():.2f}%")
        for key, part in zip(tables, evaluate(name, df, rng, base_config, start, stop)):
            tables[key].extend(part)

    pd.set_option("display.width", 250)
    for key, rows in tables.items():
        table = pd.DataFrame(rows)
        table.to_csv(OUT / f"{key}.csv", index=False, float_format="%.4f")
        print(f"\n== {key}", flush=True)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)

    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "datasets": {k: {"source": str(v[0].relative_to(REPO_ROOT)).replace("\\", "/"),
                             "rows": [v[1], v[2]],
                             **({"generator_overrides": DATASETS[k]} if k in DATASETS else {})}
                         for k, v in sources.items()},
            "start": START.isoformat(), "rows_per_generated_dataset": WEEK,
            "models": {k: [v[0], str(v[1].relative_to(REPO_ROOT))] for k, v in MODELS.items()},
            "reference": REFERENCE,
            "selection_rule": "the varied-spike model replaces v3 only if its original_val RMSE "
                              "is within 2% of v3's AND its varied_val RMSE is lower "
                              "(validation periods only; fixed before training)",
            "note": "No model or Prophet fit was refit or retuned on these datasets.",
            "runtime_seconds": round(time.time() - started, 1)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    _log(f"Done in {time.time() - started:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
