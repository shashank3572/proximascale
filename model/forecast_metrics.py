"""
ProximaScale - Evaluation plan: forecast-level metrics (M1-M4, M6, C4, D3)
---------------------------------------------------------------------------
Post-processing of Step 0's per-window prediction dump (model/eval_dump.py).
No model runs here, so every number comes from the same predictions as the
dump's Table II check and the D1 replay (model/decision_replay.py).

    table_ii.csv       Table II with the plan's corrections (Section E): +/- sd
                       over the MC seeds, lead-time distribution, net excess
                       crossings split into spurious and missed (M2), event
                       metrics (M1), the linear-extrapolation reference (C4)
                       and the generator's feasible lead-time ceiling
    events.csv         M1 at every threshold of D3's sweep (65-85%), for each
                       forecast and for the upper bound as the trigger (M4)
    spikes.csv         M1 per spike: each method's lead time next to the ceiling
    regimes.csv        M3: RMSE/MAE per regime of the target row, with the
                       generator's noise floor
    uncertainty.csv,   M4: coverage of mean + k sigma per regime, bound height,
    reliability.csv    Spearman(sigma, |error|), the k each coverage would need
    significance.csv   M6: Diebold-Mariano (HAC), moving-block bootstrap CIs for
                       dRMSE/dMAE, bootstrap over spikes for d(mean lead time)
    anomaly_sweep.csv  D3: anomaly detector precision, recall and delay for each
                       z, against the generator's known spike starts
    alpha_sweep.csv    D3: counterfactual weight vs the generator's true base load
    passes_sweep.csv   D3: MC passes N, from sibling passesN dumps if present

Definitions follow evaluate.py: threshold 75%; spike onset = first row above
it; lead time = how early the earliest forecast made 1-3 steps before the
onset exceeds it (0 if none). The script re-derives Table II from the dump
and stops unless it matches the dump's own Table II check, which was scored
with evaluate.py's helpers -- so the definitions cannot drift.

Usage (from the repo root, after model/eval_dump.py):
    python model/forecast_metrics.py
    python model/forecast_metrics.py --dump results/eval_dump/cf_off
Writes results/forecast_metrics/<dump tag>/: the CSVs above and meta.json.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

import anomaly                                     # the live detector, unmodified
from counterfactual import ACTUAL_WEIGHT, BASELINE_WEIGHT, apply_counterfactual_correction
from generate_realistic_data import SPIKE_LEN
from ground_truth import generator_ground_truth
from preprocessing import chronological_split, load_csv

DATA_CSV = REPO_ROOT / "data" / "collected" / "metrics.csv"
DEFAULT_DUMP = REPO_ROOT / "results" / "eval_dump" / "baseline"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "forecast_metrics"

THRESHOLD = 75.0                  # evaluate.py's SCALE_THRESHOLD
THRESHOLD_SWEEP = (65.0, 70.0, 75.0, 80.0, 85.0)
CROSSING_TOLERANCE = 3            # windows; persistence lags the truth by exactly the horizon
K_SWEEP = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
COVERAGE_TARGETS = (0.90, 0.95, 0.99)
# Diebold-Mariano HAC lags: h-1 (the plan's minimum), and 30 min, longer than one spike
# plus its post-scaling run (30 rows)
DM_LAGS = (2, 60)
BLOCK_LEN = 120                  # moving-block bootstrap: 1 h of windows
N_BOOT = 2000
N_BOOT_SPIKES = 10000
BOOT_SEED = 12345
Z_SWEEP = (2.0, 2.5, 3.0, 3.5)
ALPHA_SWEEP = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SPIKE_REGIMES = ("ramp", "hold", "decay")

METHODS = {                       # forecasts of CPU at t+3, in Table II order
    "persistence": "Persistence (reactive signal)",
    "linear": "Linear extrapolation (C4)",
    "prophet": "Prophet alone",
    "univariate": "Univariate LSTM",
    "hybrid": "ProximaScale (hybrid mean)",
}
TRIGGERS = {**METHODS, "hybrid_upper": "ProximaScale upper bound (mean + 2 sigma)"}
BASELINES = ("persistence", "linear", "prophet", "univariate")
REGIME_GROUPS = {
    "all": None,
    "normal": ("normal",),
    "spike": SPIKE_REGIMES,
    "ramp": ("ramp",),
    "hold": ("hold",),
    "decay": ("decay",),
    "post_scaling": ("post_scaling",),
}
TABLE_II_NAMES = {"Reactive": "persistence", "Univariate LSTM": "univariate",
                  "Prophet Alone": "prophet", "Hybrid (Full)": "hybrid"}
DEFINITIONS = {
    "event": "CPU above the threshold at the target row t+3 (90 s ahead); one per window",
    "precision, recall, f1": "window level: alarm = forecast above the threshold",
    "false_alarms_per_day": "alarm episodes (runs of consecutive alarm windows) that overlap "
                            "no window whose target row is above the threshold, per day",
    "lead time": "evaluate.py: (onset - t) x 30 s for the earliest t in [onset-3, onset-1] "
                 "whose forecast exceeds the threshold, else 0 (missed)",
    "ceiling_lead_s": "no input changes before a spike's first ramp row, so no forecast "
                      "made earlier is informed: min(onset - ramp start, 3) x 30 s",
    "crossings": "changes of (forecast > threshold) between consecutive windows "
                 "(evaluate.py's 'oscillations')",
    "spurious/missed crossings": f"forecast and true crossings matched one-to-one when they "
                                 f"have the same direction and are at most "
                                 f"{CROSSING_TOLERANCE} windows apart; unmatched forecast "
                                 "crossings are spurious, unmatched true ones missed. A forecast "
                                 "crossing that comes more than 3 windows early or late counts "
                                 "once as spurious and once as missed",
    "spike_*": "lead time over spike onsets only: the first crossing of the threshold inside "
               "each generated spike that reaches it. evaluate.py's onsets (lead_mean_s) also "
               "include normal-load noise crossings below 75% and re-crossings during a "
               "spike's hold above 75%; at exactly 75% the two coincide",
    "noise floor": "error of an oracle that knows the generator's noiseless base load "
                   "(seasonal_cpu) but not its N(0, 3^2) noise: normal rows, and post-scaling "
                   "rows (0.8 x seasonal against recorded CPU)",
    "coverage": "share of windows with actual CPU at t+3 <= mean + k sigma (one-sided)",
    "k_for_coverage": "the empirical quantile of (actual - mean) / sigma: the multiplier "
                      "that would have reached that coverage ON THE TEST SET (descriptive; "
                      "a deployed k must be chosen on validation data)",
    "diebold_mariano": "d = loss(ProximaScale) - loss(baseline); Newey-West (Bartlett) "
                       "long-run variance; negative statistic = ProximaScale better; "
                       "two-sided normal p-value",
    "block bootstrap": f"moving blocks of {BLOCK_LEN} windows, {N_BOOT} replicates, same "
                       "blocks for every method (paired), percentile 95% CI",
    "spike bootstrap": f"the spikes resampled with replacement, {N_BOOT_SPIKES} replicates",
    "seed aggregation": "mean and sd (ddof=1) over the dump's MC seeds; for CIs and "
                        "p-values, the most conservative seed",
}


# ============================================================================
# Forecasts and events
# ============================================================================

def linear_extrapolation(cpu, t_rows, horizon=3):
    """C4: the slope of the last 3 points, extended `horizon` steps, clamped to
    0-100 like predict_load(). cpu is the test split's CPU, t_rows each
    window's last row."""
    t = np.asarray(t_rows, dtype=int)
    slope = (cpu[t] - cpu[t - 2]) / 2.0
    return np.clip(cpu[t] + horizon * slope, 0.0, 100.0)


def forecasts(dump, cpu, horizon):
    """Every forecast of CPU at t+3 in one seed's dump, plus the upper bound."""
    return {
        "persistence": dump["current_cpu"].to_numpy(dtype=float),
        "linear": linear_extrapolation(cpu, dump["t"].to_numpy(), horizon),
        "prophet": dump["prophet_t3"].to_numpy(dtype=float),
        "univariate": dump["univariate_t3"].to_numpy(dtype=float),
        "hybrid": dump["predicted_load"].to_numpy(dtype=float),
        "hybrid_upper": dump["upper_bound"].to_numpy(dtype=float),
    }


def onsets_above(cpu, threshold):
    """evaluate.py's spike onsets: rows above the threshold after one at or below it."""
    above = np.asarray(cpu) > threshold
    return np.flatnonzero(above[1:] & ~above[:-1]) + 1


def lead_times(pred, t_rows, onsets, threshold, horizon, interval_s):
    """evaluate.py's lead time for each onset. t_rows must be consecutive."""
    first_t = int(t_rows[0])
    leads = np.zeros(len(onsets))
    for j, onset in enumerate(onsets):
        for t in range(onset - horizon, onset):  # earliest first
            i = t - first_t
            if 0 <= i < len(pred) and pred[i] > threshold:
                leads[j] = (onset - t) * interval_s
                break
    return leads


def lead_ceiling(onsets, spike_start_rows, horizon, interval_s):
    """Per onset: the generator's ramp start, rows from it to the onset, and
    the most lead any forecast can have without guessing (see DEFINITIONS)."""
    starts = np.sort(np.asarray(spike_start_rows))
    ramp_start = np.array([starts[starts <= onset].max() for onset in onsets])
    gap = np.asarray(onsets) - ramp_start
    return ramp_start, gap, np.minimum(gap, horizon) * interval_s


def runs(flags):
    """(starts, stops) of each run of True values; stops are exclusive."""
    padded = np.concatenate([[False], np.asarray(flags, dtype=bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2]


def crossings(flags):
    """Indices where flags changes, and whether each change is upward."""
    flags = np.asarray(flags, dtype=bool)
    idx = np.flatnonzero(flags[1:] != flags[:-1]) + 1
    return idx, flags[idx]


def count_matches(true_idx, pred_idx, tolerance):
    """Largest one-to-one matching between two sorted index lists with
    |pred - true| <= tolerance. Greedy two-pointer matching is optimal here:
    the tolerance intervals all have the same length, so they are ordered."""
    i = j = matched = 0
    while i < len(true_idx) and j < len(pred_idx):
        if pred_idx[j] < true_idx[i] - tolerance:
            j += 1
        elif pred_idx[j] > true_idx[i] + tolerance:
            i += 1
        else:
            matched, i, j = matched + 1, i + 1, j + 1
    return matched


def crossing_split(pred_flags, true_flags, tolerance=CROSSING_TOLERANCE):
    """M2: forecast crossings minus true crossings, split into spurious and missed."""
    p_idx, p_up = crossings(pred_flags)
    t_idx, t_up = crossings(true_flags)
    matched = sum(count_matches(t_idx[t_up == up], p_idx[p_up == up], tolerance)
                  for up in (True, False))
    return {"crossings": len(p_idx), "true_crossings": len(t_idx),
            "net_excess_crossings": len(p_idx) - len(t_idx),
            "spurious_crossings": len(p_idx) - matched,
            "missed_crossings": len(t_idx) - matched}


def spike_onsets_above(cpu, spike_start_rows, threshold, spike_len=SPIKE_LEN):
    """The first row above the threshold inside each generated spike that
    reaches it, and that spike's first ramp row."""
    onsets, starts = [], []
    for start in spike_start_rows:
        above = np.flatnonzero(cpu[start:start + spike_len] > threshold)
        if len(above):
            onsets.append(start + above[0])
            starts.append(start)
    return np.array(onsets, dtype=int), np.array(starts, dtype=int)


def _lead_summary(leads, prefix, horizon, interval_s):
    out = {f"{prefix}onsets": len(leads),
           f"{prefix}lead_mean_s": float(leads.mean()) if len(leads) else np.nan}
    for k in range(horizon, 0, -1):
        out[f"{prefix}caught_{k * interval_s}s"] = int(np.sum(leads == k * interval_s))
    out[f"{prefix}missed"] = int(np.sum(leads == 0))
    return out


def event_metrics(pred, actual, t_rows, onsets, threshold, horizon, interval_s,
                  spike_onsets=None):
    """M1 and M2 for one forecast. Returns (metrics, lead time per onset).
    With spike_onsets, also the lead-time summary over those (spike_*)."""
    alarm, event = pred > threshold, actual > threshold
    tp = int(np.sum(alarm & event))
    fp = int(np.sum(alarm & ~event))
    fn = int(np.sum(~alarm & event))
    starts, stops = runs(alarm)
    false_alarms = sum(1 for a, b in zip(starts, stops) if not event[a:b].any())
    leads = lead_times(pred, t_rows, onsets, threshold, horizon, interval_s)
    metrics = _lead_summary(leads, "", horizon, interval_s)
    if spike_onsets is not None:
        metrics.update(_lead_summary(
            lead_times(pred, t_rows, spike_onsets, threshold, horizon, interval_s),
            "spike_", horizon, interval_s))
    metrics.update({
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
        "alarm_episodes": len(starts),
        "false_alarms": false_alarms,
        "false_alarms_per_day": false_alarms / (len(pred) * interval_s / 86400),
    })
    metrics.update(crossing_split(alarm, event))
    return metrics, leads


# ============================================================================
# Error, uncertainty and significance
# ============================================================================

def rmse(err):
    return float(np.sqrt(np.mean(np.square(err))))


def mae(err):
    return float(np.mean(np.abs(err)))


def regime_mask(regimes, members):
    return np.ones(len(regimes), dtype=bool) if members is None else np.isin(regimes, members)


def uncertainty_metrics(actual, mean, sigma):
    """M4 for one set of windows (unclamped mean and MC sigma at t+3)."""
    err = actual - mean
    out = {"n": len(actual), "mean_sigma": float(sigma.mean()),
           "mean_bound_height": float(2 * sigma.mean()),
           "coverage_upper": float(np.mean(actual <= mean + 2 * sigma)),
           "coverage_two_sided": float(np.mean(np.abs(err) <= 2 * sigma)),
           "spearman_sigma_abs_error": float(stats.spearmanr(sigma, np.abs(err))[0])}
    z = err / np.where(sigma > 0, sigma, np.nan)
    for target in COVERAGE_TARGETS:
        out[f"k_for_{target:.0%}"] = float(np.nanquantile(z, target))
    return out


def newey_west_variance(d, lag):
    """Long-run variance of d with Bartlett weights (never negative)."""
    d = np.asarray(d, dtype=float) - np.mean(d)
    n = len(d)
    variance = d @ d / n
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return float(variance)


def diebold_mariano(loss_a, loss_b, lag):
    """DM statistic and two-sided p-value for mean(loss_a - loss_b) = 0."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    stat = d.mean() / np.sqrt(newey_west_variance(d, lag) / len(d))
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def block_bootstrap_means(values, block_len, n_boot, rng):
    """Moving-block bootstrap of row means: values (m, T) -> (n_boot, m). Every
    replicate uses the same blocks for all m rows, so differences are paired."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    m, n = values.shape
    n_blocks = int(np.ceil(n / block_len))
    cumsum = np.concatenate([np.zeros((m, 1)), np.cumsum(values, axis=1)], axis=1)
    starts = rng.integers(0, n - block_len + 1, size=(n_boot, n_blocks))
    sums = (cumsum[:, starts + block_len] - cumsum[:, starts]).sum(axis=2)
    return (sums / (n_blocks * block_len)).T


def significance_rows(fc, actual, leads, seed):
    """M6 for one seed: ProximaScale against each baseline."""
    names = list(METHODS)
    errors = np.stack([fc[name] - actual for name in names])
    boot = block_bootstrap_means(np.vstack([errors ** 2, np.abs(errors)]),
                                 BLOCK_LEN, N_BOOT, np.random.default_rng(BOOT_SEED))
    boot_rmse, boot_mae = np.sqrt(boot[:, :len(names)]), boot[:, len(names):]
    spike_idx = np.random.default_rng(BOOT_SEED + 1).integers(
        0, len(leads["hybrid"]), size=(N_BOOT_SPIKES, len(leads["hybrid"])))
    h = names.index("hybrid")
    rows = []
    for baseline in BASELINES:
        b = names.index(baseline)
        row = {"baseline": baseline, "seed": seed}
        for metric, point, draws in [("rmse", rmse, boot_rmse), ("mae", mae, boot_mae)]:
            row[f"d_{metric}"] = point(errors[h]) - point(errors[b])
            row[f"d_{metric}_lo"], row[f"d_{metric}_hi"] = np.percentile(
                draws[:, h] - draws[:, b], [2.5, 97.5])
        for loss_name, loss in [("sq", np.square), ("abs", np.abs)]:
            for lag in DM_LAGS:
                stat, p = diebold_mariano(loss(errors[h]), loss(errors[b]), lag)
                row[f"dm_{loss_name}_lag{lag}"], row[f"p_{loss_name}_lag{lag}"] = stat, p
        d_lead = leads["hybrid"] - leads[baseline]
        row["d_lead_s"] = float(d_lead.mean())
        row["d_lead_lo"], row["d_lead_hi"] = np.percentile(d_lead[spike_idx].mean(axis=1),
                                                           [2.5, 97.5])
        rows.append(row)
    return rows


def summarize_significance(per_seed):
    """Across seeds: mean of point estimates and statistics, the widest CI and
    the largest p-value (the most conservative seed)."""
    out = []
    for baseline, group in per_seed.groupby("baseline", sort=False):
        row = {"baseline": baseline, "n_seeds": len(group)}
        for col in group.columns.drop(["baseline", "seed"]):
            if col.endswith("_lo"):
                row[col] = group[col].min()
            elif col.endswith("_hi"):
                row[col] = group[col].max()
            elif col.startswith("p_"):
                row[col] = group[col].max()
            else:
                row[col] = group[col].mean()
                if col in ("d_rmse", "d_mae", "d_lead_s"):
                    row[col + "_sd"] = group[col].std(ddof=1) if len(group) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out)


# ============================================================================
# D3: anomaly detector, counterfactual weight
# ============================================================================

def window_matrix(t_rows, window):
    """Row indices of each window ending at t: shape (n, window)."""
    return np.asarray(t_rows, dtype=int)[:, None] - (window - 1) + np.arange(window)


def corrected_window_cpu(cpu, post, t_rows, window,
                         actual_weight=ACTUAL_WEIGHT, baseline_weight=BASELINE_WEIGHT):
    """Each window's CPU after predict_load()'s per-window counterfactual
    correction: a post-scaling row is blended with the last clean row before
    its run, if that row is inside the window; otherwise left raw."""
    idx = np.arange(len(cpu))
    last_clean = np.maximum.accumulate(np.where(post, -1, idx))
    rows = window_matrix(t_rows, window)
    base_row = last_clean[rows]
    blend = post[rows] & (base_row >= rows[:, :1])
    return np.where(blend, actual_weight * cpu[rows]
                    + baseline_weight * cpu[np.maximum(base_row, 0)], cpu[rows])


def latest_zscores(values):
    """anomaly.rolling_zscore() for every row of values (latest reading last)."""
    baseline, latest = values[:, :-1], values[:, -1]
    sd = baseline.std(axis=1)
    return np.where(sd < 1e-3, 0.0, (latest - baseline.mean(axis=1)) / np.maximum(sd, 1e-3))


def anomaly_flags(values, threshold, min_cpu):
    """anomaly.is_anomaly() for every row of values."""
    return (values[:, -1] >= min_cpu) & (latest_zscores(values) > threshold)


def detector_scores(flag_rows, n_positions, regimes, spike_starts, interval_s, spike_len):
    """Precision, recall and delay of anomaly flags against the generator's spikes.
    A flag is a hit when its row lies in a spike (ramp, hold or decay)."""
    flag_rows = np.asarray(flag_rows, dtype=int)
    flag_regime = regimes[flag_rows]
    hits = np.isin(flag_regime, SPIKE_REGIMES)
    delays = []
    for start in spike_starts:
        inside = flag_rows[(flag_rows >= start) & (flag_rows < start + spike_len)]
        if len(inside):
            delays.append((inside.min() - start) * interval_s)
    days = n_positions * interval_s / 86400
    false = len(flag_rows) - int(hits.sum())
    return {
        "flags": len(flag_rows),
        "flags_pct_of_decisions": 100 * len(flag_rows) / n_positions,
        "precision": float(hits.mean()) if len(flag_rows) else np.nan,
        "recall": len(delays) / len(spike_starts),
        "spikes_flagged": len(delays),
        "delay_median_s": float(np.median(delays)) if delays else np.nan,
        "delay_max_s": float(np.max(delays)) if delays else np.nan,
        "false_flags": false,
        "false_flags_per_day": false / days,
        "false_flags_normal": int(np.sum(flag_regime == "normal")),
        "false_flags_post_scaling": int(np.sum(flag_regime == "post_scaling")),
        **{f"hits_{r}": int(np.sum(flag_regime == r)) for r in SPIKE_REGIMES},
    }


def anomaly_sweep(cpu, post, t_rows, regimes, spike_starts, interval_s, window, min_cpu):
    """D3: both detectors at every z in Z_SWEEP.
    decision-time : predict_load()'s flag, the one the engine acts on (9-reading
                    baseline, counterfactual-corrected window CPU), for each window
    series        : anomaly.detect_anomalies_series() (the plan's choice: 10-reading
                    baseline, raw CPU), for each row with full history"""
    decision_values = corrected_window_cpu(cpu, post, t_rows, window)
    series_rows = np.arange(window, len(cpu))
    series_values = cpu[window_matrix(series_rows, window + 1)]
    rows = []
    for z in Z_SWEEP:
        for name, values, positions in [("decision-time", decision_values, t_rows),
                                        ("series", series_values, series_rows)]:
            flagged = positions[anomaly_flags(values, z, min_cpu)]
            rows.append({"detector": name, "z": z,
                         **detector_scores(flagged, len(positions), regimes, spike_starts,
                                           interval_s, SPIKE_LEN)})
    return pd.DataFrame(rows), decision_values, series_values


def alpha_sweep(test_df, true_cpu):
    """D3: series-level correction with weight alpha on the recorded CPU, scored
    against the generator's true (pre-drop) load on post-scaling rows."""
    post = test_df["post_scaling"].to_numpy(dtype=bool)
    rows = []
    for alpha in ALPHA_SWEEP:
        corrected = apply_counterfactual_correction(
            test_df, actual_weight=alpha, baseline_weight=1 - alpha)["cpu_percent_corrected"]
        corrected = corrected.to_numpy(dtype=float)[post]
        err = corrected - true_cpu[post]
        rows.append({"alpha": alpha, "configured": bool(np.isclose(alpha, ACTUAL_WEIGHT)),
                     "rows": int(post.sum()), "rmse": rmse(err), "mae": mae(err),
                     "bias": float(err.mean()),
                     "mean_ratio_to_true": float(np.mean(corrected / true_cpu[post]))})
    return pd.DataFrame(rows)


# ============================================================================
# Loading and provenance
# ============================================================================

def load_dumps(dump_dir, columns=None):
    files = sorted(Path(dump_dir).glob("dump_seed*.csv"),
                   key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1)))
    if not files:
        raise FileNotFoundError(f"no dump_seed*.csv in {dump_dir}; run model/eval_dump.py")
    with open(Path(dump_dir) / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    dumps = [pd.read_csv(p, usecols=columns) for p in files]
    return meta, files, dumps


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path):
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
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


def aggregate(per_seed, keys, value_cols):
    """Mean and sd (ddof=1) over seeds, columns interleaved as c, c_sd."""
    grouped = per_seed.groupby(keys, sort=False)[value_cols]
    mean, sd = grouped.mean(), grouped.std(ddof=1).fillna(0.0)
    out = mean.join(sd.add_suffix("_sd"))
    out.insert(0, "n_seeds", grouped.size())
    out = out[["n_seeds"] + [c for v in value_cols for c in (v, v + "_sd")]]
    return out.reset_index()


def check_table_ii(table, dump_meta):
    """The dump's Table II check (evaluate.py's helpers) must be reproduced."""
    columns = {"RMSE": "rmse", "MAE": "mae", "Avg Lead Time (s)": "lead_mean_s",
               "Oscillations": "crossings"}
    mismatches = []
    for expected in dump_meta["table_ii_check"]:
        row = table.set_index("method").loc[TABLE_II_NAMES[expected["Method"]]]
        for theirs, ours in columns.items():
            for suffix in ("", " sd"):
                want = expected[theirs + suffix]
                got = row[ours + ("_sd" if suffix else "")]
                if not np.isclose(got, want, rtol=0, atol=1e-6):
                    mismatches.append(f"{expected['Method']} {theirs}{suffix}: {got} vs {want}")
    if mismatches:
        raise ValueError("re-derived Table II does not match the dump's check: "
                         + "; ".join(mismatches))


def sibling_passes_dumps(dump_dir, dump_meta):
    """Dumps next to this one that differ only in the number of MC passes."""
    def fingerprint(meta):
        settings = {k: v for k, v in meta["settings"].items() if k != "n_passes"}
        artifacts = {k: v["sha256"] for k, v in meta["artifacts"].items()}
        return json.dumps([settings, artifacts, meta["data"]["sha256"]], sort_keys=True)

    mine, found = fingerprint(dump_meta), {}
    for candidate in sorted(Path(dump_dir).parent.iterdir()):
        meta_path = candidate / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if fingerprint(meta) == mine and any(candidate.glob("dump_seed*.csv")):
            found[meta["settings"]["n_passes"]] = (candidate, meta)
    return dict(sorted(found.items()))


def passes_sweep(siblings, onsets, horizon, interval_s):
    """D3: point-forecast stability across seeds, accuracy, lead time and
    coverage for each number of MC passes."""
    columns = ["seed", "t", "actual_t3", "predicted_load", "hybrid_mean_t3",
               "hybrid_std_t3", "hybrid_upper_t3"]
    rows = []
    for n_passes, (path, meta) in siblings.items():
        _, _, dumps = load_dumps(path, columns)
        actual = dumps[0]["actual_t3"].to_numpy()
        t_rows = dumps[0]["t"].to_numpy()
        per_seed = []
        for dump in dumps:
            pred = dump["predicted_load"].to_numpy()
            leads = lead_times(pred, t_rows, onsets, THRESHOLD, horizon, interval_s)
            per_seed.append({"rmse": rmse(pred - actual), "mae": mae(pred - actual),
                             "lead_mean_s": leads.mean(),
                             "crossings": len(crossings(pred > THRESHOLD)[0]),
                             "coverage_upper": np.mean(actual <= dump["hybrid_upper_t3"]),
                             "mean_sigma": dump["hybrid_std_t3"].mean()})
        per_seed = pd.DataFrame(per_seed)
        means = np.stack([d["hybrid_mean_t3"].to_numpy() for d in dumps])
        row = {"n_passes": n_passes, "n_seeds": len(dumps), "dump": _display_path(path)}
        for col in per_seed.columns:
            row[col] = per_seed[col].mean()
            row[col + "_sd"] = per_seed[col].std(ddof=1) if len(dumps) > 1 else 0.0
        row["point_sd_across_seeds"] = (float(means.std(axis=0, ddof=1).mean())
                                        if len(dumps) > 1 else np.nan)
        row["point_max_diff_across_seeds"] = float((means.max(0) - means.min(0)).max())
        row["dump_runtime_s"] = meta.get("runtime_seconds")
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Console summary
# ============================================================================

def _pm(mean, sd, digits):
    return f"{mean:.{digits}f}" + (f" +/- {sd:.{digits + 1}f}" if sd > 0 else "")


def _print_summary(table, regimes, unc, reliability, sig, anomaly_df, alpha_df, passes_df,
                   events, days):
    out = pd.DataFrame({"Method": table["label"]})
    out["RMSE"] = [_pm(m, s, 2) if pd.notna(m) else "" for m, s in zip(table.rmse, table.rmse_sd)]
    out["MAE"] = [_pm(m, s, 2) if pd.notna(m) else "" for m, s in zip(table.mae, table.mae_sd)]
    out["Lead s"] = [_pm(m, s, 1) for m, s in zip(table.lead_mean_s, table.lead_mean_s_sd)]
    out["90/60/30/miss"] = [f"{a:g}/{b:g}/{c:g}/{d:g}" for a, b, c, d in
                            zip(table.caught_90s, table.caught_60s, table.caught_30s,
                                table.missed)]
    out["Net x (spur/miss)"] = ["" if pd.isna(n) else f"{n:+g} ({s:g}/{m:g})" for n, s, m in
                                zip(table.net_excess_crossings, table.spurious_crossings,
                                    table.missed_crossings)]
    out["Prec/Rec"] = ["" if pd.isna(r) else ("n/a" if pd.isna(p) else f"{p:.2f}") + f"/{r:.2f}"
                       for p, r in zip(table.precision, table.recall)]
    out["False/day"] = ["" if pd.isna(f) else f"{f:.1f}" for f in table.false_alarms_per_day]
    print(f"\n[ProximaScale] Table II, corrected (75% event at t+90 s; {days:.2f} test days)")
    print(out.to_string(index=False))

    upper = events[(events.threshold == THRESHOLD) & (events.method == "hybrid_upper")].iloc[0]
    print(f"  Upper bound as trigger (M4): lead {upper.lead_mean_s:.1f} s, "
          f"{upper.caught_60s:g}/{upper.caught_30s:g}/{upper.missed:g} at 60/30/missed, "
          f"precision {upper.precision:.2f}, recall {upper.recall:.2f}, "
          f"{upper.false_alarms_per_day:.1f} false alarms/day")

    pivot = regimes[regimes.scored_against == "recorded"].pivot(
        index="regime", columns="method", values="rmse").reindex(
        list(REGIME_GROUPS))[list(METHODS)]
    floor = regimes[regimes.scored_against == "recorded"].groupby("regime").floor_rmse.first()
    pivot["floor"] = floor.reindex(pivot.index)
    share = regimes.groupby("regime").share_pct.first().reindex(pivot.index)
    pivot.insert(0, "% rows", share)
    print("\n[ProximaScale] M3: RMSE by regime of the target row (vs recorded CPU)")
    print(pivot.round(2).to_string())
    post_true = regimes[(regimes.regime == "post_scaling")
                        & (regimes.scored_against == "true base")].set_index("method")
    print("  post-scaling vs true base load: " + ", ".join(
        f"{m} {post_true.loc[m, 'rmse']:.2f}" for m in METHODS)
        + f", floor {post_true['floor_rmse'].iloc[0]:.2f}")

    print("\n[ProximaScale] M4: MC Dropout uncertainty (mean over seeds)")
    show = unc[["regime", "mean_sigma", "coverage_upper", "coverage_two_sided",
                "spearman_sigma_abs_error", "k_for_95%"]].copy()
    print(show.round(3).to_string(index=False))
    print("  reliability (one-sided): " + ", ".join(
        f"k={r.k:g}: {r.coverage:.1%} (nominal {r.nominal:.1%})"
        for r in reliability.itertuples()))

    print("\n[ProximaScale] M6: ProximaScale minus baseline (95% CI; DM p-values, worst seed)")
    for r in sig.itertuples():
        print(f"  vs {r.baseline:11s} dRMSE {r.d_rmse:+.3f} "
              f"[{r.d_rmse_lo:+.3f}, {r.d_rmse_hi:+.3f}]  dMAE {r.d_mae:+.3f} "
              f"[{r.d_mae_lo:+.3f}, {r.d_mae_hi:+.3f}]  "
              f"p(sq) {r.p_sq_lag2:.2g}/{r.p_sq_lag60:.2g}  "
              f"p(abs) {r.p_abs_lag2:.2g}/{r.p_abs_lag60:.2g} (lag 2/60)"
              f"  dLead {r.d_lead_s:+.1f} s [{r.d_lead_lo:+.1f}, {r.d_lead_hi:+.1f}]")

    print("\n[ProximaScale] D3: anomaly detector vs the generator's spikes")
    cols = ["detector", "z", "flags_pct_of_decisions", "precision", "recall",
            "delay_median_s", "false_flags_per_day"]
    print(anomaly_df[cols].round(3).to_string(index=False))
    print("\n[ProximaScale] D3: counterfactual weight alpha on post-scaling rows "
          "(alpha = weight on the recorded value; configured 0.6)")
    print(alpha_df.round(3).to_string(index=False))
    if passes_df is not None and len(passes_df):
        print("\n[ProximaScale] D3: MC passes")
        cols = ["n_passes", "rmse", "rmse_sd", "lead_mean_s", "lead_mean_s_sd", "crossings",
                "crossings_sd", "coverage_upper", "point_sd_across_seeds"]
        print(passes_df[cols].round(4).to_string(index=False))
    sweep = events[events.method.isin(["linear", "univariate", "hybrid", "hybrid_upper"])]
    ceiling = sweep.groupby("threshold").spike_ceiling_lead_s.first()
    sweep = sweep.pivot(index="threshold", columns="method",
                        values=["spike_lead_mean_s", "f1", "false_alarms_per_day"])
    sweep.columns = [f"{metric.replace('spike_lead_mean_s', 'lead')}:{method}"
                     for metric, method in sweep.columns]
    sweep.insert(0, "ceiling", ceiling)
    print("\n[ProximaScale] D3: threshold sweep, lead over spike onsets (mean over seeds)")
    print(sweep.round(2).to_string())


# ============================================================================
# Main
# ============================================================================

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Forecast-level metrics from the prediction dump (plan M1-M4, M6, C4, D3).")
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP,
                        help="a model/eval_dump.py output folder (default: baseline)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    started = time.time()
    dump_meta, files, dumps = load_dumps(args.dump)
    settings = dump_meta["settings"]
    if settings["table_ii_scale_threshold"] != THRESHOLD:
        raise ValueError("the dump was scored with a different threshold")
    if _sha256(DATA_CSV) != dump_meta["data"]["sha256"]:
        raise ValueError("metrics.csv changed since the dump was made; re-run model/eval_dump.py")
    interval_s, horizon = settings["sampling_interval_seconds"], settings["horizon"]
    window = settings["window_size"]

    full_df = load_csv(DATA_CSV)
    train_df, test_df = chronological_split(full_df, settings["train_frac"])
    split = len(train_df)
    truth = generator_ground_truth(full_df).iloc[split:].reset_index(drop=True)
    cpu = test_df["cpu_percent"].to_numpy(dtype=float)
    post = test_df["post_scaling"].to_numpy(dtype=bool)
    regimes_row = truth["regime"].to_numpy()

    ref = dumps[0]
    t_rows, target = ref["t"].to_numpy(dtype=int), ref["target"].to_numpy(dtype=int)
    deterministic = ["t", "target", "current_cpu", "actual_t3", "true_cpu_t3", "regime_t3",
                     "univariate_t3", "prophet_t3"]
    for path, dump in zip(files, dumps):
        if not dump[deterministic].equals(ref[deterministic]):
            raise ValueError(f"{path.name} differs from {files[0].name} outside the MC columns")
    if not np.all(np.diff(t_rows) == 1):
        raise ValueError("needs one window per row (a dump made without --limit)")
    if not (np.array_equal(ref["row_target"], split + target)
            and np.array_equal(ref["regime_t3"], regimes_row[target])
            and np.allclose(ref["true_cpu_t3"], truth["true_cpu"].to_numpy()[target])
            and np.allclose(ref["current_cpu"], cpu[t_rows])
            and np.allclose(ref["actual_t3"], cpu[target])):
        raise ValueError("the dump is not aligned with metrics.csv's test split")
    actual = ref["actual_t3"].to_numpy(dtype=float)
    regime_t3 = ref["regime_t3"].to_numpy()
    seeds = [int(d["seed"].iloc[0]) for d in dumps]
    fcs = [forecasts(d, cpu, horizon) for d in dumps]
    days = len(t_rows) * interval_s / 86400

    onsets = onsets_above(cpu, THRESHOLD)
    if len(onsets) != settings["spike_onsets_in_test_split"]:
        raise ValueError("spike onsets differ from the dump's meta")
    spike_starts = np.flatnonzero(truth["spike_start"].to_numpy())
    if len(spike_starts) != len(onsets):
        raise ValueError(f"{len(spike_starts)} generated spikes but {len(onsets)} onsets")
    ramp_start, gap, ceiling = lead_ceiling(onsets, spike_starts, horizon, interval_s)
    print(f"[ProximaScale] Forecast metrics: {len(files)} seed(s) from "
          f"{_display_path(args.dump)}, {len(t_rows)} windows, {len(onsets)} spikes "
          f"(ceiling {ceiling.mean():.1f} s)", flush=True)

    # ---- M1, M2 and D3's threshold sweep; the upper bound as a trigger (M4) ----
    event_rows, spike_rows, leads_75 = [], [], {}
    for threshold in THRESHOLD_SWEEP:
        ons = onsets if threshold == THRESHOLD else onsets_above(cpu, threshold)
        sp_onsets, sp_starts = spike_onsets_above(cpu, spike_starts, threshold)
        if threshold == THRESHOLD and not np.array_equal(sp_onsets, onsets):
            raise ValueError("at the Table II threshold every onset should open a spike")
        sp_ceiling = float(np.mean(np.minimum(sp_onsets - sp_starts, horizon) * interval_s))
        for seed, fc in zip(seeds, fcs):
            for method in TRIGGERS:
                metrics, leads = event_metrics(fc[method], actual, t_rows, ons, threshold,
                                               horizon, interval_s, sp_onsets)
                event_rows.append({"threshold": threshold, "method": method, "seed": seed,
                                   "spike_ceiling_lead_s": sp_ceiling, **metrics})
                if threshold == THRESHOLD:
                    leads_75[seed, method] = leads
    events_per_seed = pd.DataFrame(event_rows)
    event_cols = [c for c in events_per_seed.columns if c not in ("threshold", "method", "seed")]
    events = aggregate(events_per_seed, ["threshold", "method"], event_cols)
    events.insert(2, "label", events["method"].map(TRIGGERS))

    for j, onset in enumerate(onsets):
        for seed in seeds:
            for method in TRIGGERS:
                lead = leads_75[seed, method][j]
                spike_rows.append({
                    "spike": j, "onset_row": int(onset),
                    "onset_ts": test_df["timestamp"].iloc[onset].isoformat(),
                    "ramp_start_row": int(ramp_start[j]), "rows_ramp_to_onset": int(gap[j]),
                    "ceiling_lead_s": float(ceiling[j]),
                    "peak_cpu": float(cpu[onset:onset + SPIKE_LEN].max()),
                    "method": method, "seed": seed, "lead_s": lead,
                    "outcome": f"{lead:.0f}s" if lead else "missed"})

    # ---- Table II, corrected ----
    acc = pd.DataFrame([{"method": m, "seed": seed, "rmse": rmse(fc[m] - actual),
                         "mae": mae(fc[m] - actual)}
                        for seed, fc in zip(seeds, fcs) for m in METHODS])
    table = aggregate(acc, ["method"], ["rmse", "mae"]).merge(
        events[events.threshold == THRESHOLD].drop(columns=["threshold", "n_seeds", "label"]),
        on="method")
    table.insert(1, "label", table["method"].map(METHODS))
    check_table_ii(table, dump_meta)
    ceiling_row = {"method": "ceiling", "label": "Feasible ceiling (generator)", "n_seeds": 0,
                   "lead_mean_s": float(ceiling.mean()), "onsets": len(onsets), "missed": 0}
    for k in range(horizon, 0, -1):
        ceiling_row[f"caught_{k * interval_s}s"] = int(np.sum(ceiling == k * interval_s))
    table = pd.concat([table, pd.DataFrame([ceiling_row])], ignore_index=True)
    table = table.fillna({c: 0.0 for c in table.columns if c.endswith("_sd")})

    # ---- M3: error by regime, with the noise floor ----
    seasonal_t3 = truth["seasonal_cpu"].to_numpy()[target]
    true_t3 = ref["true_cpu_t3"].to_numpy(dtype=float)
    regime_rows = []
    for regime, members in REGIME_GROUPS.items():
        mask = regime_mask(regime_t3, members)
        targets = [("recorded", actual)]
        if regime == "post_scaling":
            targets.append(("true base", true_t3))
        for scored_against, y in targets:
            oracle = {("normal", "recorded"): seasonal_t3,
                      ("post_scaling", "recorded"): 0.8 * seasonal_t3,
                      ("post_scaling", "true base"): seasonal_t3}.get((regime, scored_against))
            floor_err = None if oracle is None else (y - oracle)[mask]
            for method in METHODS:
                per = [(rmse((fc[method] - y)[mask]), mae((fc[method] - y)[mask]))
                       for fc in fcs]
                per = np.array(per)
                regime_rows.append({
                    "regime": regime, "scored_against": scored_against, "method": method,
                    "rows": int(mask.sum()), "share_pct": 100 * mask.mean(),
                    "rmse": per[:, 0].mean(), "rmse_sd": per[:, 0].std(ddof=1) if len(fcs) > 1
                    else 0.0, "mae": per[:, 1].mean(),
                    "mae_sd": per[:, 1].std(ddof=1) if len(fcs) > 1 else 0.0,
                    "floor_rmse": np.nan if floor_err is None else rmse(floor_err),
                    "floor_mae": np.nan if floor_err is None else mae(floor_err)})
    regimes = pd.DataFrame(regime_rows)

    # ---- M4: uncertainty quality ----
    unc_rows, rel_rows = [], []
    for seed, dump in zip(seeds, dumps):
        mean = dump["hybrid_mean_t3"].to_numpy(dtype=float)
        sigma = dump["hybrid_std_t3"].to_numpy(dtype=float)
        if not settings.get("upper_bound_calibration") and not np.allclose(
                dump["hybrid_upper_t3"], mean + settings["std_multiplier"] * sigma, atol=1e-3):
            raise ValueError("hybrid_upper_t3 is not mean + std_multiplier x sigma")
        for regime, members in REGIME_GROUPS.items():
            mask = regime_mask(regime_t3, members)
            unc_rows.append({"regime": regime, "seed": seed,
                             **uncertainty_metrics(actual[mask], mean[mask], sigma[mask])})
        for k in K_SWEEP:
            rel_rows.append({"k": k, "seed": seed, "nominal": stats.norm.cdf(k),
                             "coverage": np.mean(actual <= mean + k * sigma),
                             **{f"coverage_{r}": np.mean((actual <= mean + k * sigma)
                                                         [regime_mask(regime_t3, REGIME_GROUPS[r])])
                                for r in ("normal", "spike", "post_scaling")}})
    unc_per_seed = pd.DataFrame(unc_rows)
    unc = aggregate(unc_per_seed, ["regime"],
                    [c for c in unc_per_seed.columns if c not in ("regime", "seed", "n")])
    unc.insert(1, "rows", unc_per_seed.groupby("regime", sort=False)["n"].first().to_numpy())
    rel_per_seed = pd.DataFrame(rel_rows)
    reliability = aggregate(rel_per_seed, ["k"],
                            [c for c in rel_per_seed.columns if c not in ("k", "seed")])

    # ---- M6: significance ----
    sig_rows = []
    for seed, fc in zip(seeds, fcs):
        leads = {m: leads_75[seed, m] for m in METHODS}
        sig_rows += significance_rows(fc, actual, leads, seed)
    sig_per_seed = pd.DataFrame(sig_rows)
    sig = summarize_significance(sig_per_seed)

    # ---- D3: anomaly z, counterfactual alpha, MC passes ----
    min_cpu = settings["anomaly_min_cpu_percent"]
    anomaly_df, decision_values, series_values = anomaly_sweep(
        cpu, post, t_rows, regimes_row, spike_starts, interval_s, window, min_cpu)
    configured_z = settings["anomaly_z_threshold"]
    if not np.array_equal(anomaly_flags(decision_values, configured_z, min_cpu),
                          ref["anomaly_flag"].to_numpy(dtype=bool)):
        raise ValueError("re-derived decision-time anomaly flags differ from the dump's")
    live_series = anomaly.detect_anomalies_series(cpu, window_size=window, threshold=configured_z)
    if not np.array_equal(anomaly_flags(series_values, configured_z, min_cpu),
                          live_series[window:]):
        raise ValueError("re-derived series flags differ from detect_anomalies_series()")
    alpha_df = alpha_sweep(test_df, truth["true_cpu"].to_numpy(dtype=float))
    siblings = sibling_passes_dumps(args.dump, dump_meta)
    passes_df = passes_sweep(siblings, onsets, horizon, interval_s) if siblings else None

    # ---- Write ----
    out_dir = args.out_dir / args.dump.name
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"table_ii.csv": table, "events.csv": events,
               "events_per_seed.csv": events_per_seed, "spikes.csv": pd.DataFrame(spike_rows),
               "regimes.csv": regimes, "uncertainty.csv": unc, "reliability.csv": reliability,
               "significance.csv": sig, "significance_per_seed.csv": sig_per_seed,
               "anomaly_sweep.csv": anomaly_df, "alpha_sweep.csv": alpha_df}
    if passes_df is not None:
        outputs["passes_sweep.csv"] = passes_df
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, index=False, float_format="%.6g")
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": [Path(sys.argv[0]).name, *(sys.argv[1:] if argv is None else argv)],
        "runtime_seconds": round(time.time() - started, 1),
        "dump": {"path": _display_path(args.dump),
                 "files": {p.name: _sha256(p) for p in files},
                 "settings": settings, "artifacts": dump_meta.get("artifacts"),
                 "git": dump_meta.get("git")},
        "passes_sweep_dumps": {str(n): _display_path(p) for n, (p, _) in siblings.items()},
        "test_split": {"windows": len(t_rows), "days": days, "spike_onsets": len(onsets),
                       "first_window_t": int(t_rows[0]), "last_window_t": int(t_rows[-1])},
        "settings": {"threshold": THRESHOLD, "threshold_sweep": THRESHOLD_SWEEP,
                     "crossing_tolerance_windows": CROSSING_TOLERANCE, "k_sweep": K_SWEEP,
                     "dm_lags": DM_LAGS, "block_len_windows": BLOCK_LEN, "n_boot": N_BOOT,
                     "n_boot_spikes": N_BOOT_SPIKES, "boot_seed": BOOT_SEED,
                     "z_sweep": Z_SWEEP, "alpha_sweep": ALPHA_SWEEP},
        "checks": ["re-derived Table II matches the dump's Table II check (evaluate.py)",
                   "dump aligned with metrics.csv and the generator's ground truth",
                   "decision-time anomaly flags at the configured z match the dump",
                   "vectorized series detector matches anomaly.detect_anomalies_series()"],
        "definitions": DEFINITIONS,
        "methods": TRIGGERS,
        "git": _git_state(),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    _print_summary(table, regimes, unc, reliability, sig, anomaly_df, alpha_df, passes_df,
                   events, days)
    print(f"\n[ProximaScale] Done in {time.time() - started:.0f}s -> {_display_path(out_dir)}",
          flush=True)
    return outputs


if __name__ == "__main__":
    main()
