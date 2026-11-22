"""
ProximaScale - Upper-bound calibration and decision-engine tuning for the v2 model
---------------------------------------------------------------------------------
Everything is CHOSEN on the validation period (the last 10% of the training
period, rows 64800-71999 of metrics.csv) and only then scored on the test
split, once.

1. Calibration. The MC Dropout bound mean + 2 sigma covers the actual t+3
   load far less often than a 2-sigma bound should. Three ways to set the
   upper bound are fit to a target one-sided coverage on validation:
       scaled     mean + k * sigma
       margin     mean + q                      (split-conformal, constant)
       binned     mean + q_b, q_b per sigma quintile (conformal, sigma-aware)
   The method with the narrowest validation bound at 95% is kept.

2. Engine. The real DecisionEngine (decision/engine.py) is replayed on the
   validation period over a small grid:
       bound       none, raw mean + 2 sigma, calibrated at 80 / 90 / 95%
       guard       scale_down_max_current_cpu: off, 60, 75 (no scale-down
                   while the current reading is above it)
       anomaly     anomaly_z_threshold 2.5 (config.yaml), 3, 3.5, 4, 5, or off
   Selection rule, fixed before looking at the results: the most spikes
   anticipated (a scale-up 30-90 s before onset), allowing one fewer than the
   best; among those, fewest false scale-ups per day; then fewest 5-minute
   reversals; then fewest replicas on average.

    python model/tune_decisions.py [--bundle DIR --test-dump DIR --out DIR]
Writes <out>/{calibration, grid_validation, chosen_test}.csv, meta.json
(default: the v2 bundle, results/eval_dump/v2, results/tuning).
"""

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decision_replay as R  # noqa: E402
import forecast_metrics as F  # noqa: E402
import train_variants as T  # noqa: E402
from preprocessing import load_csv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = REPO_ROOT / "data" / "collected" / "metrics.csv"
BUNDLE = REPO_ROOT / "model" / "saved" / "proximascale_v2"
TEST_DUMP = REPO_ROOT / "results" / "eval_dump" / "v2"
OUT = REPO_ROOT / "results" / "tuning"
CONFIG = REPO_ROOT / "config.yaml"

VAL_SEEDS = (0, 1, 2)
N_PASSES = 30
LEVELS = (0.80, 0.90, 0.95)
N_BINS = 5
GUARDS = (None, 60.0, 75.0)
ANOMALY = (2.5, 3.0, 3.5, 4.0, 5.0, None)
FLOORS = (False, True)       # scale_up_on_current_cpu (reactive floor)
INTERVAL_S = 30


def _log(message):
    print(f"[ProximaScale] {message}", flush=True)


# ============================================================================
# Calibration
# ============================================================================

def fit_bound(method, mean, sigma, actual, level):
    """Parameters of one bound so that P(actual <= bound) = level on these rows."""
    err = actual - mean
    if method == "scaled":
        return {"k": float(np.quantile(err / np.maximum(sigma, 1e-6), level))}
    if method == "margin":
        return {"q": float(np.quantile(err, level))}
    if method == "binned":
        edges = np.quantile(sigma, np.linspace(0, 1, N_BINS + 1)[1:-1])
        bins = np.digitize(sigma, edges)
        return {"edges": edges.tolist(),
                "q": [float(np.quantile(err[bins == b], level)) for b in range(N_BINS)]}
    raise ValueError(method)


def apply_bound(method, params, mean, sigma):
    if method == "raw":
        return mean + 2.0 * sigma
    if method == "scaled":
        return mean + params["k"] * sigma
    if method == "margin":
        return mean + params["q"]
    bins = np.digitize(sigma, np.asarray(params["edges"]))
    return mean + np.asarray(params["q"])[bins]


# ============================================================================
# Predictions
# ============================================================================

def validation_predictions(full_df):
    """v2's t+3 MC mean and sigma on the validation windows, one set per seed."""
    from forecaster import StackedForecaster
    from prophet_model import load_prophet_model
    start, stop = 64800, 72000
    t_abs = T.period_t_rows(start, stop, 10)
    prophet = load_prophet_model()
    X = prophet_fc = None
    out = []
    for seed in VAL_SEEDS:
        fc = StackedForecaster(BUNDLE, prophet, seed=seed)
        if X is None:
            X, prophet_fc = fc.history_inputs(full_df, t_abs)
        r = fc.forecast(X, prophet_fc, n_passes=N_PASSES)
        out.append({"seed": seed, "mean": r["mean"][:, -1], "sigma": r["sigma"][:, -1]})
        _log(f"  validation seed {seed} done")
    return t_abs, out


def period(full_df, t_abs, start, preds):
    """Everything the replay needs for one period, per seed."""
    cpu_full = full_df["cpu_percent"].to_numpy(dtype=float)
    post = full_df["post_scaling"].to_numpy(dtype=bool)
    corrected = F.corrected_window_cpu(cpu_full, post, t_abs, 10)   # predict_load's window
    z = F.latest_zscores(corrected)
    return {"t_abs": t_abs, "t": t_abs - start, "start": start,
            "cpu": cpu_full[start:t_abs[-1] + 4], "now": cpu_full[t_abs],
            "actual": cpu_full[t_abs + 3], "ts": full_df["timestamp"].iloc[t_abs]
            .dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(),
            "z": z, "latest_corrected": corrected[:, -1], "preds": preds}


def onsets_of(cpu, first_ok):
    th = R.SPIKE_THRESHOLD
    return [i for i in range(max(1, first_ok), len(cpu)) if cpu[i] > th and cpu[i - 1] <= th]


# ============================================================================
# Replay
# ============================================================================

def run(per, base_config, bound, guard, anomaly, min_cpu, floor=False):
    """Replays one setting on every seed of a period; mean of the metrics."""
    rows = []
    onsets = onsets_of(per["cpu"], per["t"][0] + 12)
    for pred in per["preds"]:
        mean, sigma = pred["mean"], pred["sigma"]
        dump = pd.DataFrame({"ts_t": per["ts"], "predicted_load": np.clip(mean, 0, 100),
                             "current_cpu": per["now"]})
        upper_col = None
        if bound is not None:
            dump["upper_bound"] = np.clip(apply_bound(bound[0], bound[1], mean, sigma), 0, 100)
            upper_col = "upper_bound"
        anomaly_col = None
        if anomaly is not None:
            dump["anomaly_flag"] = (per["latest_corrected"] >= min_cpu) & (per["z"] > anomaly)
            anomaly_col = "anomaly_flag"
        # Start from the ORIGINAL engine, whatever config.yaml now switches off:
        # each grid point sets the bound, anomaly and guard itself.
        overrides = {**R.ORIGINAL_ENGINE, "scale_down_max_current_cpu": guard,
                     "scale_up_on_current_cpu": bool(floor)}
        policy = R.Policy("tuned", "predicted_load", upper_col, anomaly_col, overrides, "",
                          current="current_cpu")
        result = R.replay(policy, dump, base_config)
        metrics, _ = R.score(result["signals"], result["reasons"], result["replicas"],
                             per["t"], per["cpu"], onsets, INTERVAL_S, 3,
                             base_config.get("max_containers", 10))
        rows.append(metrics)
    out = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    out["onsets"] = len(onsets)
    return out


def choose(grid):
    best = grid["anticipated"].max()
    ok = grid[grid["anticipated"] >= best - 1]
    return ok.sort_values(["false_ups_per_day", "reversals_5min", "mean_replicas"]).iloc[0]


def main():
    global BUNDLE, TEST_DUMP, OUT
    parser = argparse.ArgumentParser(description="Calibrate the upper bound and tune the engine on validation.")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--test-dump", type=Path, default=TEST_DUMP)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    BUNDLE, TEST_DUMP, OUT = args.bundle, args.test_dump, args.out
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    full_df = load_csv(DATA_CSV)
    with open(CONFIG, encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}
    min_cpu = float(base_config.get("anomaly_min_cpu_percent", 5.0))

    _log("v2 predictions on the validation period...")
    t_val, val_preds = validation_predictions(full_df)
    val = period(full_df, t_val, 64800, val_preds)

    test_files = sorted(TEST_DUMP.glob("dump_seed*.csv"))
    test_preds, t_test = [], None
    for path in test_files:
        d = pd.read_csv(path, usecols=["seed", "t", "hybrid_mean_t3", "hybrid_std_t3"])
        t_test = d["t"].to_numpy() + 72000
        test_preds.append({"seed": int(d["seed"].iloc[0]), "mean": d["hybrid_mean_t3"].to_numpy(),
                           "sigma": d["hybrid_std_t3"].to_numpy()})
    test = period(full_df, t_test, 72000, test_preds)

    # ---- 1. calibration
    def stacked(per):
        return (np.concatenate([p["mean"] for p in per["preds"]]),
                np.concatenate([p["sigma"] for p in per["preds"]]),
                np.tile(per["actual"], len(per["preds"])))
    vm, vs, va = stacked(val)
    tm, ts_, ta = stacked(test)
    cal_rows, fitted = [], {}
    for level in LEVELS:
        for method in ("scaled", "margin", "binned"):
            params = fit_bound(method, vm, vs, va, level)
            fitted[(method, level)] = params
            for name, (m, s, a) in (("validation", (vm, vs, va)), ("test", (tm, ts_, ta))):
                ub = apply_bound(method, params, m, s)
                spike = a > R.SPIKE_THRESHOLD
                cal_rows.append({"level": level, "method": method, "period": name,
                                 "coverage": float(np.mean(a <= ub)),
                                 "coverage_above_75": float(np.mean(a[spike] <= ub[spike])),
                                 "mean_width": float(np.mean(ub - m)), "params": json.dumps(params)})
    for name, (m, s, a) in (("validation", (vm, vs, va)), ("test", (tm, ts_, ta))):
        ub = m + 2 * s
        spike = a > R.SPIKE_THRESHOLD
        cal_rows.append({"level": None, "method": "raw (mean + 2 sigma)", "period": name,
                         "coverage": float(np.mean(a <= ub)),
                         "coverage_above_75": float(np.mean(a[spike] <= ub[spike])),
                         "mean_width": float(np.mean(ub - m)), "params": ""})
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(OUT / "calibration.csv", index=False, float_format="%.4f")
    at95 = cal[(cal.level == 0.95) & (cal.period == "validation")]
    method = at95.sort_values("mean_width").iloc[0]["method"]
    _log(f"Calibration (chosen on validation: {method}):")
    print(cal.drop(columns="params").to_string(index=False), flush=True)

    # ---- 2. engine grid on validation
    bounds = {"none": None, "raw 2 sigma": ("raw", None)}
    for level in LEVELS:
        bounds[f"{method} {int(level * 100)}%"] = (method, fitted[(method, level)])
    grid = []
    combos = list(product(bounds, GUARDS, ANOMALY, FLOORS))
    _log(f"Engine grid: {len(combos)} settings x {len(VAL_SEEDS)} seeds on validation...")
    for b, g, a, fl in combos:
        m = run(val, base_config, bounds[b], g, a, min_cpu, fl)
        grid.append({"bound": b, "guard": g, "anomaly_z": a, "floor": fl, **m})
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "grid_validation.csv", index=False, float_format="%.4f")
    chosen = choose(grid)
    _log("Chosen on validation: bound={bound}, guard={guard}, anomaly_z={anomaly_z}, "
         "floor={floor}".format(
        **chosen))

    cols = ["bound", "guard", "anomaly_z", "floor", "anticipated", "onsets", "missed", "reacted",
            "false_ups_per_day", "reversals_5min", "ups", "mean_replicas", "time_at_max_pct"]
    show = grid.sort_values(["anticipated", "false_ups_per_day"], ascending=[False, True])
    print(show[cols].head(15).to_string(index=False), flush=True)

    # ---- test, once: the chosen setting against the current ones
    current = {"bound": "raw 2 sigma", "guard": None, "anomaly_z": 2.5, "floor": False}
    settings = {
        "chosen (validation)": {k: chosen[k] for k in ("bound", "guard", "anomaly_z", "floor")},
        "tuned, guard 75, + reactive floor": {"bound": "none", "guard": 75.0, "anomaly_z": None,
                                              "floor": True},
        "current config (P4 Full)": current,
        "current, no anomaly (P3)": {**current, "anomaly_z": None},
        "forecast mean only (P2)": {"bound": "none", "guard": None, "anomaly_z": None,
                                    "floor": False},
    }
    test_rows = []
    for name, s in settings.items():
        guard = None if pd.isna(s["guard"]) else float(s["guard"])
        anomaly = None if pd.isna(s["anomaly_z"]) else float(s["anomaly_z"])
        m = run(test, base_config, bounds[s["bound"]], guard, anomaly, min_cpu, s["floor"])
        val_m = run(val, base_config, bounds[s["bound"]], guard, anomaly, min_cpu, s["floor"])
        test_rows.append({"setting": name, "bound": s["bound"], "guard": guard,
                          "anomaly_z": anomaly, "floor": bool(s["floor"]), **{f"test_{k}": v for k, v in m.items()},
                          "val_anticipated": val_m["anticipated"],
                          "val_false_ups_per_day": val_m["false_ups_per_day"]})
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(OUT / "chosen_test.csv", index=False, float_format="%.4f")
    show_cols = ["setting", "test_anticipated", "test_onsets", "test_missed", "test_reacted",
                 "test_false_ups_per_day", "test_reversals_5min", "test_ups",
                 "test_mean_replicas", "test_time_at_max_pct"]
    _log("Test split (scored once):")
    print(test_table[show_cols].to_string(index=False), flush=True)

    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "validation_rows": [64800, 72000],
            "validation_seeds": list(VAL_SEEDS), "bundle": str(BUNDLE), "test_dump": str(TEST_DUMP),
            "calibration_method": method,
            "calibration_params": {f"{lv}": fitted[(method, lv)] for lv in LEVELS},
            "chosen": {k: (None if pd.isna(chosen[k]) else chosen[k])
                       for k in ("bound", "guard", "anomaly_z", "floor")},
            "selection_rule": "max anticipated (allow best - 1), then min false_ups_per_day, "
                              "then min reversals_5min, then min mean_replicas",
            "runtime_seconds": round(time.time() - started, 1)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    _log(f"Done in {time.time() - started:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
