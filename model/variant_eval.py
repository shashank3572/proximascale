"""
ProximaScale - Evaluation D4, C1-C3, C6, D2-train and model search: scoring the retrained LSTMs
-----------------------------------------------------------------------------------------------
Scores every model from train_variants.py with the inference path
predict_load() uses: per-window counterfactual correction (left off for
variants trained without it), scaling with that run's own scalers, 30 MC
Dropout passes (mean and sigma), plus Prophet for residual targets, and the
0-100 clamp. A deterministic pass (dropout off) is scored too, because that
is how evaluate.py scores the univariate baseline.

Two periods are scored:
    validation  the last 10% of the training period (the early-stopping
                data). Model selection uses ONLY these numbers.
    test        all 17,988 test windows, the same windows as Table II. A
                variant with a longer input window reads its extra history
                from the rows before each window, as a live loop would.

Ensembles: for each variant with several training seeds, the seeds' MC
passes are pooled (N_PASSES / seeds each, so the same 30 forward passes per
decision as a single model) into one mean and sigma.

Before scoring anything, the same code is run on the committed artifacts and
must reproduce results/eval_dump/baseline (ProximaScale for MC seed 0, and the
univariate baseline) to 1e-4, so the variants and Table II share one pipeline.

C3 (conventional hybrid) is w x multi_direct + (1 - w) x Prophet, with w
chosen on the validation period.

Metrics are forecast_metrics.py's: RMSE and MAE against recorded CPU at
t+90 s, M1 lead time and false alarms at 75%, M2 crossings, M3 RMSE by
regime, and the paired block-bootstrap CI and Diebold-Mariano test of each
model against the fair ProximaScale (M6), plus lead-time differences over
the spike onsets.

    python model/variant_eval.py               # about 10 minutes
Writes results/variants/: summary.csv (test, mean +/- sd over training
seeds), validation.csv, selection.csv, per_seed.csv, significance.csv,
lead_significance.csv, blend.csv, training.csv and meta.json, and the
predictions to results/eval_dump/variants/.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import forecast_metrics as F  # noqa: E402
import train_variants as T  # noqa: E402
from ground_truth import generator_ground_truth  # noqa: E402
from preprocessing import FEATURE_COLS, chronological_split, load_csv  # noqa: E402
from train_variants import TRAIN_FRAC, VARIANTS, VARIANTS_DIR  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = REPO_ROOT / "data" / "collected" / "metrics.csv"
BASELINE_DUMP = REPO_ROOT / "results" / "eval_dump" / "baseline"
OUT_DIR = REPO_ROOT / "results" / "variants"
PRED_DIR = REPO_ROOT / "results" / "eval_dump" / "variants"
SAVED = REPO_ROOT / "model" / "saved"

HORIZON = 3
INTERVAL_S = 30
THRESHOLD = 75.0
N_PASSES = 30
MC_SEED = 0
BASE_WINDOW = 10            # fixes the set of scored windows for every variant
BLEND_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)

LABELS = {
    "persistence": "Persistence (reactive signal)",
    "prophet": "Prophet alone",
    "committed_univariate": "Univariate LSTM, committed (20 epochs, no validation)",
    "committed_hybrid": "ProximaScale, committed (stopped on test)",
    "univariate": "Univariate LSTM, fair (C6)",
    "multi_direct": "Multivariate LSTM, direct (C1)",
    "uni_residual": "Prophet + univariate residual LSTM (C2)",
    "hybrid": "ProximaScale, fair (D4)",
    "hybrid_cf_off": "ProximaScale, no correction in training or inference (D2)",
    "blend": "Weighted blend of C1 and Prophet (C3)",
    "hx_prophet_in": "ProximaScale + Prophet forecast as input",
    "hx_resid_in": "ProximaScale, residual history as input",
    "hx_clean": "ProximaScale, Prophet fit on base load only",
    "hx_window20": "ProximaScale, 20-step window",
    "hx_window30": "ProximaScale, 30-step window",
    "hx_weighted": "ProximaScale, spike-weighted loss",
    "hx_deltas": "ProximaScale + CPU and request deltas",
    "hx_wide": "ProximaScale, LSTM 128/64",
    "hx_w30_prophet": "ProximaScale, 30-step window + Prophet forecast as input",
    "hx_w30_resid": "ProximaScale, 30-step window, residual history as input",
    "hx_w60_prophet": "ProximaScale, 60-step window + Prophet forecast as input",
    "hx3_all": "v2 + residual history as an extra input",
    "hx3_huber": "v2 trained with Huber loss",
    "hx3_plateau": "v2 with learning-rate decay on plateau",
    "hx3_drop10": "v2 with dropout 0.1",
    "hx3_conv": "v2 with a causal Conv1D front end",
}
GRID_2X2 = {"univariate": ("CPU only", "direct"), "uni_residual": ("CPU only", "Prophet residual"),
            "multi_direct": ("CPU, memory, requests", "direct"),
            "hybrid": ("CPU, memory, requests", "Prophet residual")}
ENSEMBLE = "+ens"


def label(model):
    base = model.removesuffix(ENSEMBLE)
    text = LABELS.get(base, base)
    return text + (f", {len(T.DEFAULT_SEEDS)}-seed ensemble" if model.endswith(ENSEMBLE) else "")


def _log(message):
    print(f"[ProximaScale] {message}", flush=True)


# ============================================================================
# Inputs and inference (predict_load()'s path, batched across windows)
# ============================================================================

class Frame:
    """The windows of one period [start, stop) of metrics.csv: every window
    whose BASE_WINDOW inputs and t+3 target lie in the period. Inputs of a
    longer window reach back before the period (earlier history)."""

    def __init__(self, full_df, start, stop, yhats):
        self.full_cpu = full_df["cpu_percent"].to_numpy(dtype=float)
        self.full_post = full_df["post_scaling"].to_numpy(dtype=bool)
        self.arrays = {"memory_percent": full_df["memory_percent"].to_numpy(dtype=float),
                       "request_rate": full_df["request_rate"].to_numpy(dtype=float)}
        self.yhats = yhats                          # Prophet fit name -> value at every row
        self.start = start
        self.t_abs = T.period_t_rows(start, stop, BASE_WINDOW)
        self.t = self.t_abs - start                 # period-local, as eval_dump's t
        self.cpu = self.full_cpu[start:stop]        # period-local raw CPU, for onsets
        self.actual = self.full_cpu[self.t_abs + HORIZON]
        if self.t_abs[0] - 29 < 0:
            raise ValueError("period starts too early for a 30-step window")

    def inputs(self, spec):
        w = spec["window"]
        rows = T.window_rows(self.t_abs, w)
        cpu = (F.corrected_window_cpu(self.full_cpu, self.full_post, self.t_abs, w)
               if spec["correct"] else self.full_cpu[rows])
        arrays = {**self.arrays, "yhat": self.yhats[spec["prophet"]]}
        return T.build_inputs(spec["inputs"], cpu, rows, arrays)

    def prophet(self, name="committed"):
        return self.yhats[name][self.t_abs + HORIZON]


def scaled(raw, scaler):
    n, w, k = raw.shape
    return scaler.transform(raw.reshape(-1, k)).reshape(n, w, k).astype(np.float32)


def load_seeded(path, seed):
    """Keras 3 draws each Dropout layer's random state when the layer is
    built, so the seed has to be set before loading (as eval_dump.py does)."""
    import gc
    import tensorflow as tf
    tf.keras.backend.clear_session()    # 60+ models per run: free the previous ones
    gc.collect()
    tf.keras.utils.set_random_seed(seed)
    return tf.keras.models.load_model(path, compile=False)


def infer(model_path, X, target_scaler, prophet_t3, residual, n_passes=N_PASSES, seed=MC_SEED):
    """t+3 forecasts in CPU %: deterministic, the MC passes, their mean and
    sigma, and the MC upper bound mean + 2 sigma (Prophet added for residuals)."""
    model = load_seeded(model_path, seed)
    passes = np.stack([model(X, training=True).numpy()[:, -1] for _ in range(n_passes)])
    det = model.predict(X, verbose=0, batch_size=4096)[:, -1]
    scale = target_scaler.scale_[0]

    def inv(v):
        return target_scaler.inverse_transform(np.asarray(v).reshape(-1, 1))[:, 0]
    base = prophet_t3 if residual else 0.0
    mean = inv(passes.mean(axis=0)) + base
    sigma = passes.std(axis=0) / scale
    cpu_passes = (passes - target_scaler.min_[0]) / scale + base   # each pass in CPU %
    return {"det": inv(det) + base, "mc": mean, "sigma": sigma, "upper": mean + 2 * sigma,
            "passes": cpu_passes}


def run_variant(variant, seed_dir, frame, n_passes=N_PASSES):
    spec = VARIANTS[variant]
    with open(seed_dir / "feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open(seed_dir / "target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
    X = scaled(frame.inputs(spec), feature_scaler)
    return infer(seed_dir / "lstm.h5", X, target_scaler, frame.prophet(spec["prophet"]),
                 spec["target"] == "residual", n_passes=n_passes)


def pool(outs):
    """Seed ensemble: pooled MC passes -> mean, sigma and bound; mean of the
    deterministic forecasts."""
    passes = np.concatenate([o["passes"] for o in outs])
    mean, sigma = passes.mean(axis=0), passes.std(axis=0)
    return {"det": np.mean([o["det"] for o in outs], axis=0), "mc": mean, "sigma": sigma,
            "upper": mean + 2 * sigma}


def reproduce_committed(test):
    """The committed artifacts through this file's path must equal the dump."""
    dump = pd.read_csv(BASELINE_DUMP / f"dump_seed{MC_SEED}.csv",
                       usecols=["t", "hybrid_mean_t3", "hybrid_std_t3", "univariate_t3"])
    if not np.array_equal(dump["t"].to_numpy(), test.t):
        raise ValueError("baseline dump windows differ from this file's windows")
    with open(SAVED / "scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open(SAVED / "scaler_residual.pkl", "rb") as f:
        residual_scaler = pickle.load(f)
    paper = {**T.DEFAULTS, "inputs": list(FEATURE_COLS)}
    hybrid = infer(SAVED / "proximascale_lstm.h5", scaled(test.inputs(paper), feature_scaler),
                   residual_scaler, test.prophet(), residual=True)
    with open(SAVED / "scaler_univariate.pkl", "rb") as f:
        uni_scaler = pickle.load(f)
    uni = infer(SAVED / "proximascale_lstm_univariate.h5",
                scaled(test.inputs({**paper, "inputs": ["cpu_percent"]}), uni_scaler),
                uni_scaler, test.prophet(), residual=False)
    for name, mine, ref in (("ProximaScale MC mean", hybrid["mc"], dump["hybrid_mean_t3"]),
                            ("ProximaScale sigma", hybrid["sigma"], dump["hybrid_std_t3"]),
                            ("univariate", uni["det"], dump["univariate_t3"])):
        diff = float(np.max(np.abs(mine - ref.to_numpy())))
        if diff > 1e-4:
            raise ValueError(f"{name} differs from results/eval_dump/baseline by {diff:.2e}")
        _log(f"  reproduces baseline dump: {name} (max diff {diff:.1e})")
    return hybrid, uni


# ============================================================================
# Scoring
# ============================================================================

class Scorer:
    """Test-period scoring (regimes, generator truth) or validation-period
    scoring (error and the M1 event metrics only)."""

    def __init__(self, frame, truth=None, split=None):
        self.frame = frame
        self.onsets = F.onsets_above(frame.cpu, THRESHOLD)
        self.truth = truth is not None
        if self.truth:
            target_rows = frame.t_abs + HORIZON
            self.regimes = truth["regime"].to_numpy()[target_rows]
            self.true_cpu = truth["true_cpu"].to_numpy()[target_rows]
            spike_rows = np.flatnonzero(truth["spike_start"].to_numpy()[split:])
            self.spike_onsets, _ = F.spike_onsets_above(frame.cpu, spike_rows, THRESHOLD)
            _, _, ceiling = F.lead_ceiling(self.onsets, spike_rows, HORIZON, INTERVAL_S)
            self.ceiling_lead = float(ceiling.mean())

    def __call__(self, pred):
        f = self.frame
        pred = np.clip(pred, 0.0, 100.0)
        err = pred - f.actual
        metrics, _ = F.event_metrics(pred, f.actual, f.t, self.onsets, THRESHOLD, HORIZON,
                                     INTERVAL_S,
                                     spike_onsets=self.spike_onsets if self.truth else None)
        row = {"rmse": F.rmse(err), "mae": F.mae(err), **metrics}
        if self.truth:
            row["lead_gap_to_ceiling_s"] = self.ceiling_lead - metrics["lead_mean_s"]
            for name in ("normal", "spike", "ramp", "hold", "decay", "post_scaling"):
                mask = F.regime_mask(self.regimes, F.REGIME_GROUPS[name])
                row[f"rmse_{name}"] = F.rmse(err[mask])
            post = self.regimes == "post_scaling"
            row["rmse_post_scaling_true"] = F.rmse(pred[post] - self.true_cpu[post])
        return row

    def leads(self, pred):
        f = self.frame
        return F.lead_times(np.clip(pred, 0.0, 100.0), f.t, self.onsets, THRESHOLD, HORIZON,
                            INTERVAL_S)


def paired(pred_a, pred_b, actual, rng):
    """Model a minus model b: dRMSE with a block-bootstrap 95% CI, dMAE, and
    DM p-values on squared and absolute error."""
    ea, eb = np.clip(pred_a, 0, 100) - actual, np.clip(pred_b, 0, 100) - actual
    boot = F.block_bootstrap_means(np.vstack([ea ** 2, eb ** 2, np.abs(ea), np.abs(eb)]),
                                   F.BLOCK_LEN, F.N_BOOT, rng)
    d_rmse = np.sqrt(boot[:, 0]) - np.sqrt(boot[:, 1])
    d_mae = boot[:, 2] - boot[:, 3]
    out = {"d_rmse": F.rmse(ea) - F.rmse(eb), "d_rmse_lo": np.percentile(d_rmse, 2.5),
           "d_rmse_hi": np.percentile(d_rmse, 97.5), "d_mae": F.mae(ea) - F.mae(eb),
           "d_mae_lo": np.percentile(d_mae, 2.5), "d_mae_hi": np.percentile(d_mae, 97.5)}
    for lag in F.DM_LAGS:
        out[f"p_sq_lag{lag}"] = F.diebold_mariano(ea ** 2, eb ** 2, lag)[1]
        out[f"p_abs_lag{lag}"] = F.diebold_mariano(np.abs(ea), np.abs(eb), lag)[1]
    return out


def lead_differences(leads, reference, rng, n_boot=F.N_BOOT_SPIKES):
    """Lead time of `reference` minus each other model, per spike onset and
    averaged over training seeds, with a bootstrap 95% CI over the onsets."""
    ref = np.mean(leads[reference], axis=0)
    rows = []
    for name, per_seed in leads.items():
        if name == reference:
            continue
        d = ref - np.mean(per_seed, axis=0)
        boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
        rows.append({"reference": reference, "versus": name, "d_lead_s": d.mean(),
                     "d_lead_lo": np.percentile(boot, 2.5), "d_lead_hi": np.percentile(boot, 97.5)})
    return rows


def tune_blend(val, c1_val):
    """C3 weight on the validation period only."""
    rows = [{"w": w, "val_rmse": F.rmse(np.clip(w * c1_val + (1 - w) * val.prophet(), 0, 100)
                                        - val.actual)} for w in BLEND_GRID]
    table = pd.DataFrame(rows)
    return float(table.loc[table["val_rmse"].idxmin(), "w"]), table


def summarize(per_seed):
    value_cols = [c for c in per_seed.columns
                  if c not in ("model", "label", "forecast", "train_seed", "forecast_kind")]
    keys = ["model", "label", "forecast_kind"]
    summary = per_seed.groupby(keys, sort=False)[value_cols].agg(["mean", "std"])
    summary.columns = [f"{c}{'' if s == 'mean' else '_sd'}" for c, s in summary.columns]
    summary = summary.reset_index()
    summary.insert(3, "train_seeds", per_seed.groupby(keys, sort=False).size().to_numpy())
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score the retrained LSTMs.")
    parser.add_argument("--variants-dir", type=Path, default=VARIANTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    started = time.time()

    import tensorflow as tf
    tf.keras.utils.disable_interactive_logging()

    full_df = load_csv(DATA_CSV)
    train_df, _ = chronological_split(full_df, TRAIN_FRAC)
    split = len(train_df)
    val_start = int(split * (1 - T.VAL_FRAC))
    truth = generator_ground_truth(full_df)

    runs = sorted((v, int(d.name.removeprefix("seed")), d)
                  for v in VARIANTS if (args.variants_dir / v).is_dir()
                  for d in (args.variants_dir / v).glob("seed*") if (d / "info.json").exists())
    if not runs:
        raise FileNotFoundError(f"no trained runs in {args.variants_dir}; run train_variants.py")
    prophets = sorted({"committed"} | {VARIANTS[v]["prophet"] for v, _, _ in runs})
    yhats = {name: T.prophet_yhat(name, full_df["timestamp"]) for name in prophets}
    test = Frame(full_df, split, len(full_df), yhats)
    val = Frame(full_df, val_start, split, yhats)
    score_test = Scorer(test, truth, split)
    score_val = Scorer(val)
    _log(f"{len(test.t)} test windows, {len(score_test.onsets)} onsets (ceiling "
         f"{score_test.ceiling_lead:.1f} s); validation {len(val.t)} windows, "
         f"{len(score_val.onsets)} onsets")

    _log("Checking this pipeline against the committed models' dump...")
    committed_hybrid, committed_uni = reproduce_committed(test)

    rows, val_rows, sig_rows, blend_rows, training_rows = [], [], [], [], []

    def emit(model, kind, seed, test_pred, val_pred=None):
        base = {"model": model, "label": label(model), "forecast": kind, "train_seed": seed}
        rows.append({**base, **score_test(test_pred)})
        if val_pred is not None:
            val_rows.append({**base, **score_val(val_pred)})

    emit("persistence", "deterministic", -1, test.full_cpu[test.t_abs],
         val.full_cpu[val.t_abs])
    emit("prophet", "deterministic", -1, test.prophet(), val.prophet())
    emit("committed_univariate", "deterministic", -1, committed_uni["det"])
    emit("committed_hybrid", "MC mean", -1, committed_hybrid["mc"])
    emit("committed_hybrid", "MC upper bound", -1, committed_hybrid["upper"])

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(F.BOOT_SEED)
    leads, results = {}, {}
    kinds = (("deterministic", "det"), ("MC mean", "mc"), ("MC upper bound", "upper"))
    for variant, seed, seed_dir in runs:
        _log(f"{variant} seed {seed}")
        out_test = run_variant(variant, seed_dir, test)
        out_val = run_variant(variant, seed_dir, val)
        results[(variant, seed)] = out_test
        for kind, key in kinds:
            emit(variant, kind, seed, out_test[key], out_val[key])
        leads.setdefault(variant, []).append(score_test.leads(out_test["mc"]))
        if variant == "hybrid":
            leads.setdefault("hybrid_upper", []).append(score_test.leads(out_test["upper"]))
        info = json.loads((seed_dir / "info.json").read_text())
        training_rows.append({k: info[k] for k in (
            "variant", "seed", "epochs_run", "best_epoch", "best_val_loss",
            "train_loss_at_best", "train_seconds", "parameters")})
        if variant == "multi_direct":
            w, table = tune_blend(val, out_val["mc"])
            blend_rows.append(table.assign(train_seed=seed, chosen=table["w"] == w))
            blend = w * out_test["mc"] + (1 - w) * test.prophet()
            blend_val = w * out_val["mc"] + (1 - w) * val.prophet()
            emit("blend", f"MC mean, w={w:.2f}", seed, blend, blend_val)
            results[("blend", seed)] = {"mc": blend}
            leads.setdefault("blend", []).append(score_test.leads(blend))

    # Seed ensembles: the same 30 forward passes per decision, split across seeds.
    by_variant = {}
    for variant, seed, seed_dir in runs:
        by_variant.setdefault(variant, []).append(seed_dir)
    for variant, dirs in by_variant.items():
        if len(dirs) < 2:
            continue
        _log(f"{variant}: {len(dirs)}-seed ensemble")
        n = N_PASSES // len(dirs)
        ens_test = pool([run_variant(variant, d, test, n_passes=n) for d in dirs])
        ens_val = pool([run_variant(variant, d, val, n_passes=n) for d in dirs])
        name = variant + ENSEMBLE
        for kind, key in kinds:
            emit(name, kind, -2, ens_test[key], ens_val[key])
        results[(name, -2)] = ens_test
        leads[name] = [score_test.leads(ens_test["mc"])]

    # M6 against the fair ProximaScale, paired by training seed (ensembles
    # against the fair ProximaScale ensemble).
    for (name, seed), res in results.items():
        if name in ("hybrid", "hybrid" + ENSEMBLE):
            continue
        ref = results.get(("hybrid" + ENSEMBLE, -2) if seed == -2 else ("hybrid", seed))
        if ref is not None:
            sig_rows.append({"train_seed": seed, "versus": name,
                             **paired(ref["mc"], res["mc"], test.actual, rng)})
    for seed in sorted({s for _, s, _ in runs}):
        ref = results.get(("hybrid", seed))
        if ref is None:
            continue
        for name, pred in (("committed_hybrid", committed_hybrid["mc"]),
                           ("persistence", test.full_cpu[test.t_abs]),
                           ("prophet", test.prophet())):
            sig_rows.append({"train_seed": seed, "versus": name,
                             **paired(ref["mc"], pred, test.actual, rng)})

    preds = pd.DataFrame({"t": test.t, "actual_t3": test.actual,
                          "true_cpu_t3": score_test.true_cpu, "regime_t3": score_test.regimes,
                          "prophet_t3": test.prophet(),
                          **{f"{n}_s{s}_mc_t3": r["mc"] for (n, s), r in results.items()},
                          **{f"{n}_s{s}_upper_t3": r["upper"] for (n, s), r in results.items()
                             if "upper" in r}})
    preds.to_csv(PRED_DIR / "predictions.csv", index=False, float_format="%.4f")

    per_seed, val_seed = pd.DataFrame(rows), pd.DataFrame(val_rows)
    for frame in (per_seed, val_seed):
        frame["forecast_kind"] = frame["forecast"].str.replace(r", w=.*", "", regex=True)
    summary, val_summary = summarize(per_seed), summarize(val_seed)

    # Selection, on validation only: Prophet + LSTM models by validation RMSE
    # of the MC mean (the forecast predict_load() returns).
    residual_models = {v for v in VARIANTS if VARIANTS[v]["target"] == "residual"}
    is_candidate = val_summary["model"].str.removesuffix(ENSEMBLE).isin(residual_models)
    selection = (val_summary[is_candidate & (val_summary["forecast_kind"] == "MC mean")]
                 .sort_values("rmse")[["model", "label", "train_seeds", "rmse", "mae",
                                       "lead_mean_s", "missed", "false_alarms_per_day"]]
                 .rename(columns=lambda c: c if c in ("model", "label", "train_seeds")
                         else f"val_{c}"))
    test_cols = summary[summary["forecast_kind"] == "MC mean"][
        ["model", "rmse", "mae", "lead_mean_s", "missed", "false_alarms_per_day",
         "net_excess_crossings"]].rename(columns=lambda c: c if c == "model" else f"test_{c}")
    selection = selection.merge(test_cols, on="model", how="left")

    sig = pd.DataFrame(sig_rows)
    sig_summary = sig.groupby("versus", sort=False).agg(
        d_rmse=("d_rmse", "mean"), d_rmse_sd=("d_rmse", "std"), d_rmse_lo=("d_rmse_lo", "min"),
        d_rmse_hi=("d_rmse_hi", "max"), d_mae=("d_mae", "mean"), d_mae_lo=("d_mae_lo", "min"),
        d_mae_hi=("d_mae_hi", "max"),
        **{c: (c, "max") for c in sig.columns if c.startswith("p_")}).reset_index()
    lead_sig = pd.DataFrame([row for ref in ("hybrid", "hybrid" + ENSEMBLE) if ref in leads
                             for row in lead_differences(leads, ref, rng)])

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    per_seed.drop(columns="forecast_kind").to_csv(out / "per_seed.csv", index=False,
                                                  float_format="%.4f")
    summary.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    val_summary.to_csv(out / "validation.csv", index=False, float_format="%.4f")
    selection.to_csv(out / "selection.csv", index=False, float_format="%.4f")
    sig.to_csv(out / "significance_per_seed.csv", index=False, float_format="%.6g")
    sig_summary.to_csv(out / "significance.csv", index=False, float_format="%.6g")
    lead_sig.to_csv(out / "lead_significance.csv", index=False, float_format="%.4f")
    training = pd.DataFrame(training_rows)
    training.to_csv(out / "training.csv", index=False, float_format="%.6f")
    if blend_rows:
        pd.concat(blend_rows).to_csv(out / "blend.csv", index=False, float_format="%.4f")
    (out / "meta.json").write_text(json.dumps({
        "mc_passes": N_PASSES, "mc_seed": MC_SEED, "threshold": THRESHOLD,
        "onsets": len(score_test.onsets), "ceiling_lead_s": score_test.ceiling_lead,
        "test_windows": len(test.t), "validation_windows": len(val.t),
        "validation_onsets": len(score_val.onsets), "selection": "validation RMSE, MC mean",
        "runs": [str(d.relative_to(REPO_ROOT)) for _, _, d in runs],
        "runtime_s": round(time.time() - started, 1),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))

    show = summary[summary["forecast_kind"] == "MC mean"]
    cols = ["label", "train_seeds", "rmse", "rmse_sd", "mae", "lead_mean_s", "missed",
            "net_excess_crossings", "false_alarms_per_day", "rmse_hold", "rmse_decay",
            "rmse_post_scaling_true"]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_colwidth", 70):
        print("\n[ProximaScale] Test, MC mean (mean and sd over training seeds)")
        print(show[cols].round(3).to_string(index=False))
        print("\n[ProximaScale] Selection on VALIDATION (Prophet + LSTM models), test alongside")
        print(selection.round(3).to_string(index=False))
        print("\n[ProximaScale] Fair ProximaScale minus each model (test)")
        print(sig_summary.round(4).to_string(index=False))
        print("\n[ProximaScale] Lead time differences over the spike onsets (95% CI)")
        print(lead_sig.round(2).to_string(index=False))
    _log(f"Done in {time.time() - started:.0f}s -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
