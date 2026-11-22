"""
ProximaScale - Evaluation Step 0: Per-Window Prediction Dump
-------------------------------------------------------------
Runs every method once over ALL test windows and saves one row per window,
so the evaluation plan's analyses (event metrics, regime-stratified error,
uncertainty coverage, significance tests, the decision-engine replay) are
post-processing of one shared set of predictions: every table then comes
from the same model outputs and the same code path.

Reused unchanged from evaluate.py: the chronological split, the window
builder, the univariate-LSTM and Prophet-alone baselines, and the metric
definitions. The "Table II check" printed at the end is evaluate.py's own
scoring, run on every window with the full MC Dropout passes.

The hybrid mirrors predict_load() step for step (counterfactual correction
-> feature scaling -> Prophet forecast for the window's own future
timestamps -> LSTM residual with MC Dropout -> Prophet + residual -> 0-100
clamp -> anomaly check), but runs each MC pass over all windows in one
batch, and keeps what predict_load() throws away: the unclamped mean, the
MC sigma and all three horizon steps. tests/test_eval_dump.py pins this
batched copy to predict_load() itself.

Seeding: Keras 3 fixes each Dropout layer's random state when the layer is
built, so each seed re-seeds Python/NumPy/TF and THEN reloads the LSTM.
The same seed always gives the same file.

Ground truth: generate_realistic_data.generate() would overwrite
metrics.csv, so its seeded RNG calls are replayed in memory instead
(model/ground_truth.py). The rebuilt CPU column must match metrics.csv
exactly, or the dump stops.

Usage (from the repo root):
    python model/eval_dump.py                        # 5 seeds x 30 passes
    python model/eval_dump.py --no-counterfactual    # plan D2: correction off
    python model/eval_dump.py --n-passes 10 --seeds 0 1 2
    python model/eval_dump.py --bundle model/saved/proximascale_v2 --tag v2
        (the v2 model, model/forecaster.py: each window reads its 30 readings
        of history from metrics.csv, as the live loop does; the anomaly flag
        and the baselines are unchanged)
Writes results/eval_dump/<tag>/dump_seed<N>.csv, one row per test window,
plus meta.json (settings, artifact hashes, git state, column guide and the
Table II check).
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF's native startup logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate as ev                  # Table II's helpers and metric definitions
import generate_realistic_data as gen  # constants only -- generate() is never called
from anomaly import _configured_min_cpu, _configured_threshold, is_anomaly
from counterfactual import apply_counterfactual_correction
from ground_truth import generator_ground_truth  # the seeded generator, replayed
from lstm_model import HORIZON, MODEL_PATH
from predict import RESIDUAL_SCALER_PATH, SAMPLING_INTERVAL_SECONDS
from preprocessing import (FEATURE_COLS, SCALER_PATH, WINDOW_SIZE, chronological_split,
                           load_csv, load_scaler, scale_features)
from prophet_model import PROPHET_PATH, get_prophet_fitted, load_prophet_model
from uncertainty import N_PASSES, STD_MULTIPLIER

assert HORIZON == 3, "the *_t3 column names below assume a 3-step horizon"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = REPO_ROOT / "data" / "collected" / "metrics.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "eval_dump"
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
STEPS = range(1, HORIZON + 1)

ARTIFACT_DEFAULTS = {
    "lstm": MODEL_PATH,
    "feature_scaler": SCALER_PATH,
    "residual_scaler": RESIDUAL_SCALER_PATH,
    "prophet": PROPHET_PATH,
    "univariate_lstm": ev.UNIVARIATE_MODEL_PATH,
    "univariate_scaler": ev.UNIVARIATE_SCALER_PATH,
}

# Output columns, in file order, with what each one means (also saved to meta.json).
COLUMNS = {
    "seed": "MC Dropout seed this file was made with",
    "window_start": "first of the 10 input rows (test-split row index)",
    "t": "last input row, i.e. 'now' (test-split row index, as in evaluate.py)",
    "target": "row being forecast, t+3, about 90 s ahead (test-split row index)",
    "row_target": "the target row's index in the full metrics.csv",
    "ts_t": "timestamp of row t",
    "ts_target": "timestamp of the target row",
    "current_cpu": "raw CPU at t; also the persistence ('Reactive') forecast",
    **{f"actual_t{k}": f"raw CPU at t+{k}" + (" (what Table II scores against)" if k == 3 else "")
       for k in STEPS},
    "true_cpu_t3": "generator's CPU at t+3 before its post-scaling 0.8x drop "
                   "(equals actual_t3 on every other row)",
    "regime_t3": "generator regime of the target row: normal, ramp, hold, decay or post_scaling",
    "post_scaling_in_window": "whether any of the 10 input rows is post_scaling",
    "univariate_t3": "univariate LSTM forecast for t+3 (evaluate.py's baseline)",
    **{f"prophet_t{k}": f"Prophet-alone forecast for t+{k}" for k in STEPS},
    **{f"hybrid_mean_t{k}": f"ProximaScale forecast for t+{k}: "
                            "Prophet + MC-mean residual, unclamped"
       for k in STEPS},
    **{f"hybrid_std_t{k}": f"MC Dropout sigma for t+{k} in CPU points "
                           "(population sd over passes, as uncertainty.py computes it)"
       for k in STEPS},
    "hybrid_upper_t3": f"hybrid_mean_t3 + {STD_MULTIPLIER:g} x hybrid_std_t3, unclamped",
    "predicted_load": "predict_load() output 1: hybrid_mean_t3 clamped to 0-100",
    "upper_bound": "predict_load() output 2: hybrid_upper_t3 clamped to 0-100",
    "anomaly_flag": "predict_load() output 3",
}


def _log(message):
    print(f"[ProximaScale] {message}", flush=True)


# ============================================================================
# predict_load(), batched across windows
# ============================================================================

def build_windows(test_df, indices, counterfactual=True):
    """evaluate.py's window dicts. counterfactual=False is plan D2: every
    post_scaling flag is cleared, so the correction leaves CPU untouched."""
    windows = [ev._row_to_window_dicts(test_df, int(i), WINDOW_SIZE) for i in indices]
    if not counterfactual:
        for window in windows:
            for row in window:
                row["post_scaling"] = False
    return windows


def anomaly_settings():
    """(z threshold, CPU floor). predict_load() resolves these from
    config.yaml on every call; here they are resolved once."""
    return _configured_threshold(), _configured_min_cpu()


def preprocess_windows(windows, feature_scaler, anomaly_threshold, anomaly_min_cpu):
    """predict.prepare_scaled_window() plus predict_load()'s anomaly check,
    one window at a time (the correction's baseline is found per window).
    Returns the scaled LSTM inputs (n, 10, features) and the anomaly flags."""
    X = np.empty((len(windows), WINDOW_SIZE, len(FEATURE_COLS)), dtype=np.float32)
    anomaly = np.empty(len(windows), dtype=bool)
    for j, window in enumerate(windows):
        df = pd.DataFrame(window)
        if "post_scaling" not in df.columns:
            df["post_scaling"] = False
        df = apply_counterfactual_correction(df)
        df["cpu_percent"] = df["cpu_percent_corrected"]
        X[j] = scale_features(df, feature_scaler)
        anomaly[j] = is_anomaly(df["cpu_percent"].values, threshold=anomaly_threshold,
                                min_cpu_percent=anomaly_min_cpu)
    return X, anomaly


def prophet_forecast(prophet_model, windows):
    """Prophet's forecast for each window's own next HORIZON timestamps,
    shape (n, HORIZON). Prophet's yhat depends only on the timestamp, so each
    distinct timestamp is predicted once (in sorted order, which is the order
    Prophet returns rows in) and mapped back to every window that needs it."""
    last_ts = pd.DatetimeIndex([pd.Timestamp(w[-1]["timestamp"]) for w in windows])
    step = pd.Timedelta(seconds=SAMPLING_INTERVAL_SECONDS)
    future = np.stack([(last_ts + step * k).values for k in STEPS], axis=1)
    unique_ts, inverse = np.unique(future.ravel(), return_inverse=True)
    yhat = np.asarray(get_prophet_fitted(prophet_model, unique_ts), dtype=float)
    return yhat[inverse].reshape(future.shape)


def load_lstm(path, seed):
    """Seeds Python, NumPy and TF, then builds the model. The order matters:
    Keras 3 draws each Dropout layer's random state when the layer is created,
    so seeding after loading would leave the MC Dropout masks unseeded."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    return tf.keras.models.load_model(path, compile=False)


def mc_hybrid(model, X, residual_scaler, prophet_fc, n_passes):
    """predict_with_uncertainty() + predict_load()'s stacking, with each MC
    pass run over every window in one batch. Returns (mean, sigma, upper),
    each shape (n, HORIZON) in CPU %, before predict_load()'s 0-100 clamp."""
    passes = np.stack([model(X, training=True).numpy() for _ in range(n_passes)])
    mean_scaled, std_scaled = passes.mean(axis=0), passes.std(axis=0)
    mean_resid = residual_scaler.inverse_transform(mean_scaled.reshape(-1, 1))
    mean_resid = mean_resid.reshape(mean_scaled.shape)
    std_resid = std_scaled / residual_scaler.scale_[0]
    mean = prophet_fc + mean_resid
    upper = prophet_fc + (mean_resid + STD_MULTIPLIER * std_resid)
    return mean, std_resid, upper


def clamp(values):
    """predict_load()'s physical 0-100 CPU% clamp."""
    return np.clip(values, 0.0, 100.0)


def load_univariate(model_path, scaler_path):
    """evaluate.py's cached univariate baseline. Its train-if-missing branch
    is deliberately not used: a missing artifact is an error here."""
    from tensorflow.keras.models import load_model
    model = load_model(model_path, compile=False)
    with open(scaler_path, "rb") as f:
        return model, pickle.load(f)


# ============================================================================
# Table II check (evaluate.py's metrics, every window)
# ============================================================================

def table_ii_check(dumps, test_df):
    """Scores each method exactly as evaluate.py does, on every window.
    Hybrid figures are the mean (and sd) over the seeds."""
    threshold = ev.SCALE_THRESHOLD
    onsets = ev._detect_spike_onsets(test_df["cpu_percent"].to_numpy(), threshold)
    ref = dumps[0]
    t_values = ref["t"].tolist()
    target_by_t = dict(zip(t_values, ref["target"].tolist()))
    actual = ref["actual_t3"].to_numpy()

    def score(pred):
        pred = np.asarray(pred, dtype=float)
        pred_by_t = dict(zip(t_values, pred))
        leads = [ev._lead_time_for_method(pred_by_t, target_by_t, onset, threshold)
                 for onset in onsets]
        return {
            "RMSE": float(np.sqrt(np.mean((pred - actual) ** 2))),
            "MAE": float(np.mean(np.abs(pred - actual))),
            "Avg Lead Time (s)": float(np.mean(leads)) if leads else 0.0,
            "Oscillations": float(ev._oscillation_count(pred, threshold)),
        }

    methods = [
        ("Reactive", [score(ref["current_cpu"])]),
        ("Univariate LSTM", [score(ref["univariate_t3"])]),
        ("Prophet Alone", [score(ref["prophet_t3"])]),
        ("Hybrid (Full)", [score(d["predicted_load"]) for d in dumps]),
    ]
    records = []
    for method, scores in methods:
        scores = pd.DataFrame(scores)
        record = {"Method": method}
        for col in scores.columns:
            record[col] = float(scores[col].mean())
            record[col + " sd"] = float(scores[col].std(ddof=1)) if len(scores) > 1 else 0.0
        records.append(record)
    return pd.DataFrame(records), len(onsets)


def _format_check(table):
    out = pd.DataFrame({"Method": table["Method"]})
    for col, digits in [("RMSE", 2), ("MAE", 2), ("Avg Lead Time (s)", 1), ("Oscillations", 1)]:
        out[col] = [f"{m:.{digits}f}" + (f" +/- {s:.{digits + 1}f}" if s > 0 else "")
                    for m, s in zip(table[col], table[col + " sd"])]
    return out.to_string(index=False)


# ============================================================================
# Provenance
# ============================================================================

def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path):
    try:
        return Path(os.path.relpath(path, REPO_ROOT)).as_posix()
    except ValueError:  # a different drive on Windows
        return str(path)


def _git_state():
    def git(*cmd):
        return subprocess.run(["git", *cmd], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout
    try:
        changed = git("status", "--porcelain", "--untracked-files=no").splitlines()
        return {"commit": git("rev-parse", "HEAD").strip(),
                "uncommitted_changes": [line[3:] for line in changed]}
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions():
    versions = {"python": sys.version.split()[0]}
    for package in ["tensorflow", "keras", "prophet", "numpy", "pandas", "scikit-learn"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


# ============================================================================
# Main
# ============================================================================

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Per-window prediction dump for the evaluation plan (Step 0).")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="MC Dropout seeds, one output file each (default: 0 1 2 3 4)")
    parser.add_argument("--n-passes", type=int, default=N_PASSES,
                        help=f"MC Dropout passes per window (default: {N_PASSES}, as predict_load)")
    parser.add_argument("--no-counterfactual", action="store_true",
                        help="plan D2: clear post_scaling in every input window")
    parser.add_argument("--limit", type=int,
                        help="only the first N windows (smoke run; not comparable to Table II)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bundle", type=Path,
                        help="serve the hybrid from a v2 bundle (model/forecaster.py) "
                             "instead of --lstm / --feature-scaler / --residual-scaler")
    parser.add_argument("--tag",
                        help="output subfolder (default: from the options, e.g. baseline, cf_off)")
    artifacts = parser.add_argument_group(
        "model artifacts (default: the committed ones in model/saved/)")
    for name, default in ARTIFACT_DEFAULTS.items():
        artifacts.add_argument("--" + name.replace("_", "-"), type=Path, default=default)

    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct")
    if args.n_passes < 1:
        parser.error("--n-passes must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    args.custom_artifacts = [name for name, default in ARTIFACT_DEFAULTS.items()
                             if getattr(args, name).resolve() != Path(default).resolve()]
    if (args.custom_artifacts or args.bundle) and not args.tag:
        parser.error("name the output folder with --tag when overriding model artifacts")
    if not args.tag:
        parts = []
        if args.no_counterfactual:
            parts.append("cf_off")
        if args.n_passes != N_PASSES:
            parts.append(f"passes{args.n_passes}")
        if args.limit is not None:
            parts.append(f"first{args.limit}")
        args.tag = "_".join(parts) or "baseline"
    return args


def main(argv=None):
    args = _parse_args(argv)
    started = time.time()
    out_dir = args.out_dir / args.tag

    full_df = load_csv(DATA_CSV)
    train_df, test_df = chronological_split(full_df, ev.TRAIN_FRAC)
    split_idx = len(train_df)
    truth = generator_ground_truth(full_df)
    _log(f"Ground truth rebuilt from the seed-{gen.SEED} generator; matches metrics.csv exactly")

    n_windows = len(test_df) - WINDOW_SIZE - HORIZON + 1
    indices = np.arange(n_windows if args.limit is None else min(args.limit, n_windows))
    windows = build_windows(test_df, indices, counterfactual=not args.no_counterfactual)
    _log(f"{len(windows)} of {n_windows} test windows | seeds {args.seeds} | "
         f"{args.n_passes} MC passes | counterfactual {'OFF' if args.no_counterfactual else 'on'}"
         f" -> {_display_path(out_dir)}")

    feature_scaler = load_scaler(args.feature_scaler)
    with open(args.residual_scaler, "rb") as f:
        residual_scaler = pickle.load(f)
    prophet_model = load_prophet_model(args.prophet)
    uni_model, uni_scaler = load_univariate(args.univariate_lstm, args.univariate_scaler)
    anomaly_threshold, anomaly_min_cpu = anomaly_settings()

    _log("Preprocessing windows (per-window counterfactual correction, as predict_load does)...")
    X, anomaly = preprocess_windows(windows, feature_scaler, anomaly_threshold, anomaly_min_cpu)
    _log("Prophet forecasts...")
    prophet_fc = prophet_forecast(prophet_model, windows)
    _log("Univariate LSTM baseline...")
    univariate = ev._univariate_predict_batch(uni_model, uni_scaler, windows)

    t = indices + WINDOW_SIZE - 1
    target = t + HORIZON
    cpu = test_df["cpu_percent"].to_numpy(dtype=float)
    stamps = test_df["timestamp"]
    flags = np.concatenate([[0], np.cumsum(test_df["post_scaling"].to_numpy(dtype=int))])
    base = pd.DataFrame({
        "window_start": indices,
        "t": t,
        "target": target,
        "row_target": split_idx + target,
        "ts_t": stamps.iloc[t].dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(),
        "ts_target": stamps.iloc[target].dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(),
        "current_cpu": [ev._reactive_predict(w) for w in windows],
        **{f"actual_t{k}": cpu[t + k] for k in STEPS},
        "true_cpu_t3": truth["true_cpu"].to_numpy()[split_idx + target],
        "regime_t3": truth["regime"].to_numpy()[split_idx + target],
        "post_scaling_in_window": flags[indices + WINDOW_SIZE] > flags[indices],
        "univariate_t3": univariate,
        **{f"prophet_t{k}": prophet_fc[:, k - 1] for k in STEPS},
    })

    X_v2 = None
    if args.bundle:
        from forecaster import StackedForecaster
        _log(f"v2 inputs from {_display_path(args.bundle)} (history read from metrics.csv)...")
        probe = StackedForecaster(args.bundle, prophet_model)
        X_v2, prophet_v2 = probe.history_inputs(full_df, split_idx + t)
        if not args.no_counterfactual and not probe.correct:
            raise ValueError("bundle config has correct=false")
        diff = float(np.max(np.abs(prophet_v2 - prophet_fc)))
        if diff > 1e-6:
            raise ValueError(f"v2 Prophet forecast differs from the dump's by {diff:.2e}")
        del probe

    out_dir.mkdir(parents=True, exist_ok=True)
    dumps = []
    for seed in args.seeds:
        seed_started = time.time()
        if args.bundle:
            out = StackedForecaster(args.bundle, prophet_model, seed=seed).forecast(
                X_v2, prophet_fc, n_passes=args.n_passes)
            mean, std, upper = out["mean"], out["sigma"], out["upper"]
        else:
            model = load_lstm(args.lstm, seed)
            if tuple(model.input_shape[1:]) != X.shape[1:]:
                raise ValueError(f"{args.lstm} expects inputs {model.input_shape[1:]}, "
                                 f"but the feature scaler produces {X.shape[1:]}")
            mean, std, upper = mc_hybrid(model, X, residual_scaler, prophet_fc, args.n_passes)
        dump = base.copy()
        dump["seed"] = seed
        for k in STEPS:
            dump[f"hybrid_mean_t{k}"] = mean[:, k - 1]
        for k in STEPS:
            dump[f"hybrid_std_t{k}"] = std[:, k - 1]
        dump["hybrid_upper_t3"] = upper[:, -1]
        dump["predicted_load"] = clamp(mean[:, -1])
        dump["upper_bound"] = clamp(upper[:, -1])
        dump["anomaly_flag"] = anomaly
        dump = dump[list(COLUMNS)]
        dump.to_csv(out_dir / f"dump_seed{seed}.csv", index=False, float_format="%.6g")
        dumps.append(dump)
        _log(f"  seed {seed}: dump_seed{seed}.csv written ({time.time() - seed_started:.0f}s)")

    check, n_onsets = table_ii_check(dumps, test_df)
    note = " -- partial test split (--limit), NOT comparable to Table II" if args.limit else ""
    _log(f"Table II check: evaluate.py's metrics on {len(windows)} windows, "
         f"{n_onsets} spike onsets, Hybrid = mean +/- sd over {len(dumps)} seed(s){note}")
    print(_format_check(check), flush=True)

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": [Path(sys.argv[0]).name, *(sys.argv[1:] if argv is None else argv)],
        "runtime_seconds": round(time.time() - started, 1),
        "settings": {
            "seeds": args.seeds,
            "n_passes": args.n_passes,
            "std_multiplier": STD_MULTIPLIER,
            # a v2 bundle with a calibrated bound (config.json "upper_bound"):
            # hybrid_upper_t3 / upper_bound are that bound, not mean + 2 sigma
            "upper_bound_calibration": (None if not args.bundle else json.loads(
                (Path(args.bundle) / "config.json").read_text()).get("upper_bound")),
            "counterfactual_at_inference": not args.no_counterfactual,
            "anomaly_z_threshold": anomaly_threshold,
            "anomaly_min_cpu_percent": anomaly_min_cpu,
            "table_ii_scale_threshold": ev.SCALE_THRESHOLD,
            "train_frac": ev.TRAIN_FRAC,
            "window_size": WINDOW_SIZE,
            "horizon": HORIZON,
            "sampling_interval_seconds": SAMPLING_INTERVAL_SECONDS,
            "windows": len(windows),
            "windows_in_test_split": n_windows,
            "spike_onsets_in_test_split": n_onsets,
        },
        "data": {"path": _display_path(DATA_CSV), "sha256": _sha256(DATA_CSV),
                 "rows": len(full_df), "test_rows": len(test_df)},
        "artifacts": {name: {"path": _display_path(getattr(args, name)),
                             "sha256": _sha256(getattr(args, name))}
                      for name in ARTIFACT_DEFAULTS},
        "custom_artifacts": args.custom_artifacts,
        "bundle": None if not args.bundle else {
            "path": _display_path(args.bundle),
            "files": {_display_path(f): _sha256(f) for f in sorted(Path(args.bundle).rglob("*"))
                      if f.is_file()}},
        "git": _git_state(),
        "versions": _versions(),
        "columns": COLUMNS,
        "table_ii_check": check.to_dict("records"),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _log(f"Done in {time.time() - started:.0f}s -> {_display_path(out_dir)}")
    return dumps, check


if __name__ == "__main__":
    main()
