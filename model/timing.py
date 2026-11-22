"""
ProximaScale - Evaluation M5: computational overhead
----------------------------------------------------
Times the live decision path one window at a time, the way main.py's loop
calls it (no batching across windows):

    predict_load(window, n_passes=N)         end to end, warm
      = Prophet forecast for the window's 3 future timestamps
      + counterfactual correction and feature scaling (prepare_scaled_window)
      + N MC Dropout passes and inverse scaling (predict_with_uncertainty)
      + the anomaly check (is_anomaly)

for N in {1, 5, 10, 30}, plus the univariate-LSTM baseline (correction,
scaling, one deterministic forward pass), the cold start (first call, which
loads the artifacts), model sizes, and the SHAP attribution main.py runs on
every scale-up (_compute_shap: training-data background + KernelExplainer,
nsamples=256).

Timings depend on the machine; meta.json records it. Run it on an otherwise
idle machine:
    python model/timing.py                     # 300 windows, about 2 minutes
    python model/timing.py --windows 100 --shap-calls 3

Writes results/timing/<tag>/: latency.csv (per component and N: mean, p50,
p95, max in ms, and the share of the 30 s sampling interval), sizes.csv and
meta.json.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))   # main.py and dashboard/ for the SHAP timing

DEFAULT_OUT_DIR = REPO_ROOT / "results" / "timing"
PASSES_SWEEP = (1, 5, 10, 30)
N_WARMUP = 10


def summarize(times_s, interval_s):
    """Latency summary in milliseconds plus the share of the sampling interval."""
    ms = np.asarray(times_s, dtype=float) * 1000.0
    return {"calls": len(ms), "mean_ms": ms.mean(), "p50_ms": np.percentile(ms, 50),
            "p95_ms": np.percentile(ms, 95), "max_ms": ms.max(),
            "pct_of_interval": 100.0 * ms.mean() / (interval_s * 1000.0)}


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def _log(message):
    print(f"[ProximaScale] {message}", flush=True)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Evaluation M5: per-decision latency.")
    parser.add_argument("--windows", type=int, default=300,
                        help="test windows to time, evenly spaced over the test split")
    parser.add_argument("--shap-calls", type=int, default=5,
                        help="scale-up windows to time SHAP on (0 skips SHAP)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="baseline")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    out_dir = args.out_dir / args.tag

    import evaluate as ev
    import predict as P
    from anomaly import is_anomaly
    from preprocessing import WINDOW_SIZE, chronological_split, load_csv
    from lstm_model import HORIZON
    from prophet_model import get_prophet_fitted
    from uncertainty import predict_with_uncertainty
    import tensorflow as tf

    interval = P.SAMPLING_INTERVAL_SECONDS
    full_df = load_csv(P.TRAINING_DATA_HINT_PATH)
    _, test_df = chronological_split(full_df, ev.TRAIN_FRAC)
    n_total = len(test_df) - WINDOW_SIZE - HORIZON + 1
    indices = np.linspace(0, n_total - 1, args.windows).astype(int)
    windows = [ev._row_to_window_dicts(test_df, int(i), WINDOW_SIZE) for i in indices]
    _log(f"Timing {len(windows)} of {n_total} test windows, one call at a time")

    # Cold start: the first predict_load() call loads every artifact.
    tf.keras.utils.set_random_seed(0)
    (cold_pred, _, _), cold_s = _timed(P.predict_load, windows[0])
    _log(f"Cold start (first predict_load call, loads artifacts): {cold_s:.2f} s")
    for window in windows[:N_WARMUP]:
        P.predict_load(window)

    rows = []
    step = pd.Timedelta(seconds=interval)
    for n_passes in PASSES_SWEEP:
        parts = {"prophet": [], "correct_scale": [], "mc_passes": [], "anomaly": [],
                 "predict_load": []}
        for window in windows:
            # Components, in predict_load()'s order and with its arguments.
            last_ts = pd.Timestamp(window[-1]["timestamp"])
            future = [last_ts + step * i for i in range(1, HORIZON + 1)]
            _, t = _timed(get_prophet_fitted, P._prophet_model, future)
            parts["prophet"].append(t)
            (scaled, corrected), t = _timed(P.prepare_scaled_window, window)
            parts["correct_scale"].append(t)
            x = scaled.reshape(1, WINDOW_SIZE, -1)
            _, t = _timed(predict_with_uncertainty, P._model, x, P._residual_scaler,
                          inverse_transform_fn=lambda v: P._residual_scaler.inverse_transform(
                              v.reshape(-1, 1)).flatten(),
                          n_passes=n_passes)
            parts["mc_passes"].append(t)
            _, t = _timed(is_anomaly, corrected)
            parts["anomaly"].append(t)
            # And the real thing, end to end.
            _, t = _timed(P.predict_load, window, n_passes=n_passes)
            parts["predict_load"].append(t)
        for component, times in parts.items():
            rows.append({"method": "ProximaScale", "n_passes": n_passes,
                         "component": component, **summarize(times, interval)})
        end = summarize(parts["predict_load"], interval)
        _log(f"N={n_passes:>2}: predict_load mean {end['mean_ms']:.1f} ms, "
             f"p95 {end['p95_ms']:.1f} ms ({end['pct_of_interval']:.3f}% of {interval} s)")

    # Univariate LSTM baseline: correction, scaling, one deterministic pass.
    uni_model, uni_scaler = ev._load_or_train_univariate_model() \
        if not ev.UNIVARIATE_MODEL_PATH.exists() else _load_univariate(ev)
    uni_times = []
    for window in windows[:N_WARMUP] + windows:
        start = time.perf_counter()
        corrected = ev._prep_corrected(window)["cpu_percent_corrected"].values
        x = uni_scaler.transform(corrected.reshape(-1, 1)).reshape(1, WINDOW_SIZE, 1)
        pred = uni_model(x, training=False).numpy()[:, -1].reshape(-1, 1)
        uni_scaler.inverse_transform(pred)
        uni_times.append(time.perf_counter() - start)
    uni = summarize(uni_times[N_WARMUP:], interval)
    rows.append({"method": "Univariate LSTM", "n_passes": 1, "component": "end_to_end", **uni})
    _log(f"Univariate LSTM: mean {uni['mean_ms']:.1f} ms, p95 {uni['p95_ms']:.1f} ms")

    # SHAP as main.py runs it on a scale-up: _compute_shap() rebuilds its
    # training-data background on every call, then runs KernelExplainer.
    shap_rows = []
    if args.shap_calls > 0:
        import main as live
        import model.predict as live_predict   # main.py's own import path and cache
        live_predict._load_artifacts()         # the live loop calls predict_load() first
        from dashboard import shap_explain
        busy = [w for w in windows if w[-1]["cpu_percent"] > 70] or windows
        picks = [busy[i] for i in np.linspace(0, len(busy) - 1, args.shap_calls).astype(int)]
        explains = []
        original_explain = shap_explain.ShapExplainer.explain

        def timed_explain(self, x):        # times KernelExplainer inside the live call
            out, t = _timed(original_explain, self, x)
            explains.append(t)
            return out

        shap_explain.ShapExplainer.explain = timed_explain
        try:
            live._compute_shap(picks[0])                   # warm-up (imports shap)
            explains.clear()
            totals = []
            for window in picks:
                attribution, t = _timed(live._compute_shap, window)
                if attribution is None:
                    raise RuntimeError("main._compute_shap returned None; see the warning above")
                totals.append(t)
        finally:
            shap_explain.ShapExplainer.explain = original_explain
        for component, times in (("shap_compute_shap", totals), ("shap_explain_only", explains)):
            shap_rows.append({"method": "ProximaScale", "n_passes": 0,
                              "component": component, **summarize(times, interval)})
        _log(f"SHAP per scale-up (main._compute_shap): mean {np.mean(totals):.2f} s; "
             f"KernelExplainer alone {np.mean(explains):.2f} s")

    latency = pd.DataFrame(rows + shap_rows)
    sizes = pd.DataFrame([
        {"artifact": "residual LSTM (.h5)", "parameters": int(P._model.count_params()),
         "bytes": Path(P.MODEL_PATH).stat().st_size},
        {"artifact": "univariate LSTM (.h5)", "parameters": int(uni_model.count_params()),
         "bytes": Path(ev.UNIVARIATE_MODEL_PATH).stat().st_size},
        {"artifact": "Prophet (.pkl)", "parameters": None,
         "bytes": Path(P.SAVED_DIR / "proximascale_prophet.pkl").stat().st_size},
    ])

    out_dir.mkdir(parents=True, exist_ok=True)
    latency.to_csv(out_dir / "latency.csv", index=False, float_format="%.4f")
    sizes.to_csv(out_dir / "sizes.csv", index=False)
    meta = {
        "windows": len(windows), "warmup_calls": N_WARMUP, "passes_sweep": list(PASSES_SWEEP),
        "sampling_interval_s": interval, "cold_start_s": cold_s,
        "shap_calls": args.shap_calls, "shap_nsamples": 256,
        "machine": {"processor": platform.processor(), "logical_cpus": os.cpu_count(),
                    "os": platform.platform(), "python": platform.python_version(),
                    "tensorflow": tf.__version__},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print()
    show = latency.pivot_table(index="component", columns="n_passes", values="mean_ms")
    _log("Mean latency (ms) by component and MC passes (0 = SHAP)")
    print(show.round(2).to_string())
    print()
    print(sizes.to_string(index=False))
    _log(f"Done -> {out_dir.relative_to(REPO_ROOT)}")


def _load_univariate(ev):
    import pickle
    from tensorflow.keras.models import load_model
    model = load_model(ev.UNIVARIATE_MODEL_PATH, compile=False)
    with open(ev.UNIVARIATE_SCALER_PATH, "rb") as f:
        return model, pickle.load(f)


if __name__ == "__main__":
    main()
