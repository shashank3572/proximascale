"""
ProximaScale - Evaluation D1: Decision-Engine Replay with Component Ablation
-----------------------------------------------------------------------------
Feeds Step 0's per-window predictions (model/eval_dump.py), one 30 s
decision at a time, into the real DecisionEngine from decision/engine.py,
and scores the scaling ACTIONS -- the decision-level evaluation that
Table II (forecast accuracy only) does not cover. Seven policies switch
the engine's components on one at a time (evaluation plan D1 and C5):

    P0 Reactive                 current CPU, no bound, no anomaly flag
    P1 Univariate LSTM          the CPU-only forecast
    P2 ProximaScale mean        the hybrid forecast alone
    P3 + upper bound            + MC upper bound (the upper_bound_risk branch)
    P4 Full                     + anomaly flag (the cooldown-bypassing branch)
    P5 Full, static thresholds  P4 with the adaptive thresholds switched off
    P6 Full, no cooldown        P4 with cooldown_seconds = 0
    P7, P8 (beyond the plan)    P5 and P6 built on P3 instead of P4, so the
                                thresholds and cooldown are isolated without
                                the anomaly branch (which dominates P4-P6)

Nothing in decision/ is modified. The engine runs as-is, with:
  - a replica counter in place of DockerActuator: same min/max bounds, same
    contract (scale_up/scale_down return False at the limit),
  - a simulated clock: each decision happens at its window's own data
    timestamp, so the 180 s cooldown is exactly 6 decisions, as in the live
    loop (main.py polls every 30 s),
  - scaling events captured in memory: log_scaling_event() rewrites its whole
    JSON file on every event and would write to data/scaling_events.json.

Open loop: scaling actions do not change the replayed data, so this scores
decision quality on a fixed trace, not closed-loop dynamics.

Usage (from the repo root, after model/eval_dump.py):
    python model/decision_replay.py                                # baseline dump
    python model/decision_replay.py --dump results/eval_dump/cf_off
Writes results/d1_replay/<dump tag>/: summary.csv (one row per policy, mean
and sd over the dump's seeds), per_seed.csv, spikes.csv (one row per policy,
seed and spike), actions.csv (every scaling action) and meta.json.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import decision.engine as engine_mod          # the real engine, unmodified
import decision.hysteresis as hysteresis_mod
from preprocessing import chronological_split, load_csv

DATA_CSV = REPO_ROOT / "data" / "collected" / "metrics.csv"
DEFAULT_DUMP = REPO_ROOT / "results" / "eval_dump" / "baseline"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "d1_replay"
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

SPIKE_THRESHOLD = 75.0     # evaluate.py's SCALE_THRESHOLD: onset = first row above it
FALSE_UP_SECONDS = 300     # a scale-up is false if CPU stays <= 75% for the next 5 min
REVERSAL_SECONDS = 300     # opposite actions at most 5 min apart count as one oscillation
CAPACITY_LOOKBACK_SECONDS = 300
DUMP_COLUMNS = ["seed", "t", "ts_t", "current_cpu", "univariate_t3",
                "predicted_load", "upper_bound", "anomaly_flag"]


@dataclass
class Policy:
    name: str
    predicted: str            # dump column fed to evaluate() as predicted_cpu
    upper: str | None         # dump column fed as upper_bound (None: not passed)
    anomaly: str | None       # dump column fed as anomaly_flag (None: always False)
    overrides: dict           # applied on top of config.yaml
    isolates: str
    current: str | None = None  # dump column fed as current_cpu (None: not passed)


_FULL = dict(predicted="predicted_load", upper="upper_bound", anomaly="anomaly_flag")
_NO_ANOMALY = dict(predicted="predicted_load", upper="upper_bound", anomaly=None)
POLICIES = [
    Policy("P0 Reactive", "current_cpu", None, None, {}, "baseline: act on the current reading"),
    Policy("P1 Univariate LSTM", "univariate_t3", None, None, {}, "a CPU-only forecaster"),
    Policy("P2 ProximaScale mean", "predicted_load", None, None, {}, "the hybrid forecaster alone"),
    Policy("P3 + upper bound", "predicted_load", "upper_bound", None, {},
           "the upper_bound_risk branch"),
    Policy("P4 Full", **_FULL, overrides={}, isolates="the anomaly bypass"),
    Policy("P5 Full, static thresholds", **_FULL, overrides={"adaptive_min_samples": 10 ** 9},
           isolates="the adaptive thresholds (P4 vs P5)"),
    Policy("P6 Full, no cooldown", **_FULL, overrides={"cooldown_seconds": 0},
           isolates="hysteresis and cooldown (P4 vs P6)"),
    Policy("P7 P3, static thresholds", **_NO_ANOMALY, overrides={"adaptive_min_samples": 10 ** 9},
           isolates="beyond the plan: adaptive thresholds without the anomaly branch (P3 vs P7)"),
    Policy("P8 P3, no cooldown", **_NO_ANOMALY, overrides={"cooldown_seconds": 0},
           isolates="beyond the plan: cooldown without the anomaly branch (P3 vs P8)"),
    Policy("P9 Tuned (config.yaml)", **_FULL, overrides={}, current="current_cpu",
           isolates="config.yaml as is: switches and guard tuned on validation "
                    "(model/tune_decisions.py)"),
]
# P0-P8 are the plan's ablation of the ORIGINAL engine: every branch on, no
# scale-down guard, whatever config.yaml now says. Only P9 uses those keys.
ORIGINAL_ENGINE = {"use_anomaly_bypass": True, "use_upper_bound_risk": True,
                   "scale_down_max_current_cpu": None, "scale_up_on_current_cpu": False}

METRICS = {
    "ups": "successful scale-ups",
    "downs": "successful scale-downs",
    "ups_anomaly": "scale-ups from the anomaly branch (bypasses cooldown)",
    "ups_upper_bound": "scale-ups from the upper_bound_risk branch",
    "ups_cpu_high": "scale-ups because the forecast was above the upper threshold",
    "ups_current_cpu": "scale-ups because the CURRENT reading was above it (reactive floor)",
    "blocked_at_max": "decisions that wanted a scale-up while already at max replicas",
    "reversals_5min": "consecutive actions in opposite directions at most 5 min apart",
    "anticipated": "spikes with a scale-up 30-90 s before onset (inside the forecast horizon)",
    "lead_median_s": "median lead over anticipated spikes (onset minus earliest such scale-up)",
    "lead_mean_s": "mean lead over anticipated spikes",
    "reacted": "spikes whose first scale-up came at or after onset, while CPU was above 75%",
    "delay_median_s": "median reaction delay over reacted spikes (first scale-up minus onset)",
    "at_max": "spikes with no scale-up because replicas were already at max_containers",
    "missed": "spikes with no scale-up from 90 s before onset to the end of the spike, "
              "although replicas were below max_containers",
    "capacity_up_before_onset": "spikes with more replicas just before onset than 5 min earlier",
    "false_ups": "scale-ups with CPU at or below 75% for all of the next 5 min",
    "false_ups_per_day": "false_ups per day of replayed trace",
    "mean_replicas": "average replica count over all decisions",
    "replicas_at_onset": "average replica count just before each spike onset",
    "time_at_max_pct": "share of decisions made at max_containers, in %",
}


class ReplicaCounter:
    """Stands in for DockerActuator: the same min/max bounds and the same
    contract (scale_up/scale_down return False at the limit and change
    nothing), with a counter instead of containers."""

    def __init__(self, min_replicas=1, max_replicas=10):
        self.min_replicas, self.max_replicas = min_replicas, max_replicas
        self.replicas = min_replicas

    def replica_count(self):
        return self.replicas

    def scale_up(self):
        if self.replicas >= self.max_replicas:
            return False
        self.replicas += 1
        return True

    def scale_down(self):
        if self.replicas <= self.min_replicas:
            return False
        self.replicas -= 1
        return True

    def hold(self):
        return True


class SimClock:
    """Takes the place of the `time` module inside decision.engine and
    decision.hysteresis: time() returns the current decision's timestamp."""

    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now


def replay(policy, dump, base_config):
    """Runs one policy over one dump through the real DecisionEngine, one
    decision per row. Returns per-decision arrays (signal, reason, replicas
    after the decision) plus one record per successful scaling action."""
    original = {} if policy.current else ORIGINAL_ENGINE
    config = {**base_config, **original, **policy.overrides}
    n = len(dump)
    predicted = dump[policy.predicted].to_numpy(dtype=float)
    upper = dump[policy.upper].to_numpy(dtype=float) if policy.upper else None
    anomaly = (dump[policy.anomaly].to_numpy(dtype=bool) if policy.anomaly
               else np.zeros(n, dtype=bool))
    current = dump[policy.current].to_numpy(dtype=float) if policy.current else None
    stamps = pd.to_datetime(dump["ts_t"])
    clock_times = (stamps - pd.Timestamp(0)).dt.total_seconds().to_numpy()

    clock = SimClock()
    actuator = ReplicaCounter(config.get("min_containers", 1), config.get("max_containers", 10))
    events = []
    signals = np.empty(n, dtype=object)
    reasons = np.empty(n, dtype=object)
    replicas = np.empty(n, dtype=int)
    actions = []
    with patch.object(engine_mod, "time", clock), patch.object(hysteresis_mod, "time", clock), \
            patch.object(engine_mod, "log_scaling_event", events.append):
        engine = engine_mod.DecisionEngine(config=config, actuator=actuator)
        for i in range(n):
            clock.now = clock_times[i]
            bound = None if upper is None else float(upper[i])
            now_cpu = None if current is None else float(current[i])
            signals[i] = engine.evaluate(float(predicted[i]), bound, anomaly_flag=bool(anomaly[i]),
                                         current_cpu=now_cpu)
            reasons[i] = engine.last_reason
            replicas[i] = actuator.replicas
            if signals[i] != "hold":
                event = events[-1]
                actions.append({
                    "row": i, "action": signals[i], "reason": reasons[i],
                    "replicas_after": replicas[i], "predicted_cpu": predicted[i],
                    "upper_bound": bound, "anomaly_flag": bool(anomaly[i]),
                    "threshold": event.get("upper_thresh", event.get("lower_thresh")),
                })
    return {"signals": signals, "reasons": reasons, "replicas": replicas, "actions": actions}


def score(signals, reasons, replicas, t_rows, cpu, onsets, interval_s, horizon_steps=3,
          max_replicas=10):
    """M7 metrics for one replay. t_rows: the trace row each decision was
    made at (consecutive); cpu: the actual CPU of every trace row; onsets:
    rows where CPU first rises above SPIKE_THRESHOLD. Each spike is
    anticipated, reacted, at_max or missed. Returns (metrics dict ordered as
    METRICS, list of per-spike records)."""
    false_rows = FALSE_UP_SECONDS // interval_s
    reversal_rows = REVERSAL_SECONDS // interval_s
    lookback_rows = CAPACITY_LOOKBACK_SECONDS // interval_s
    ups, downs = signals == "scale_up", signals == "scale_down"
    up_rows = t_rows[ups]
    above = cpu > SPIKE_THRESHOLD

    def replicas_before(row):
        """Replicas in effect when `row` arrives: after the decision at row - 1."""
        i = row - 1 - t_rows[0]
        if not 0 <= i < len(replicas):
            raise ValueError(f"row {row} has no decision before it in the replayed trace")
        return int(replicas[i])

    false_ups = int(sum(not above[r + 1:r + 1 + false_rows].any() for r in up_rows))
    acted = np.flatnonzero(ups | downs)
    directions = np.where(ups[acted], 1, -1)
    flips = (directions[1:] != directions[:-1]) & (np.diff(t_rows[acted]) <= reversal_rows)

    spikes = []
    for onset in onsets:
        below = np.flatnonzero(~above[onset:])
        end = onset + int(below[0]) if below.size else len(cpu)
        early = up_rows[(up_rows >= onset - horizon_steps) & (up_rows < onset)]
        late = up_rows[(up_rows >= onset) & (up_rows < end)]
        at_onset, earlier = replicas_before(onset), replicas_before(onset - lookback_rows)
        lead = delay = np.nan
        if early.size:
            outcome, lead = "anticipated", float((onset - early.min()) * interval_s)
        elif late.size:
            outcome, delay = "reacted", float((late.min() - onset) * interval_s)
        else:
            outcome = "at_max" if at_onset >= max_replicas else "missed"
        spikes.append({"onset_row": int(onset), "outcome": outcome, "lead_s": lead,
                       "delay_s": delay, "replicas_at_onset": at_onset,
                       "capacity_up_before_onset": at_onset > earlier})

    leads = [s["lead_s"] for s in spikes if s["outcome"] == "anticipated"]
    delays = [s["delay_s"] for s in spikes if s["outcome"] == "reacted"]
    days = len(t_rows) * interval_s / 86400
    metrics = {
        "ups": int(ups.sum()),
        "downs": int(downs.sum()),
        "ups_anomaly": int(np.sum(ups & (reasons == "anomaly"))),
        "ups_upper_bound": int(np.sum(ups & (reasons == "upper_bound_risk"))),
        "ups_cpu_high": int(np.sum(ups & (reasons == "cpu_high"))),
        "ups_current_cpu": int(np.sum(ups & (reasons == "current_cpu_high"))),
        "blocked_at_max": int(np.sum(reasons == "max_reached")),
        "reversals_5min": int(flips.sum()),
        "anticipated": len(leads),
        "lead_median_s": float(np.median(leads)) if leads else np.nan,
        "lead_mean_s": float(np.mean(leads)) if leads else np.nan,
        "reacted": len(delays),
        "delay_median_s": float(np.median(delays)) if delays else np.nan,
        "at_max": sum(s["outcome"] == "at_max" for s in spikes),
        "missed": sum(s["outcome"] == "missed" for s in spikes),
        "capacity_up_before_onset": sum(s["capacity_up_before_onset"] for s in spikes),
        "false_ups": false_ups,
        "false_ups_per_day": false_ups / days,
        "mean_replicas": float(replicas.mean()),
        "replicas_at_onset": float(np.mean([s["replicas_at_onset"] for s in spikes])),
        "time_at_max_pct": float(100 * np.mean(replicas >= max_replicas)),
    }
    assert list(metrics) == list(METRICS)
    return metrics, spikes


# ============================================================================
# Main
# ============================================================================

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


def _format_summary(summary, n_spikes):
    columns = [("ups", "ups", 0), ("reversals_5min", "flips<=5m", 0),
               ("anticipated", f"ahead/{n_spikes}", 1), ("lead_median_s", "lead s", 0),
               ("reacted", "reacted", 0), ("at_max", "at max", 1), ("missed", "missed", 0),
               ("false_ups_per_day", "false/day", 1), ("mean_replicas", "replicas", 2),
               ("time_at_max_pct", "%@max", 1)]
    out = pd.DataFrame({"policy": summary["policy"]})
    for key, label, digits in columns:
        out[label] = [("-" if pd.isna(m) else f"{m:.{digits}f}")
                      + (f" +/-{s:.{digits + 1}f}" if pd.notna(s) and s > 0 else "")
                      for m, s in zip(summary[key], summary[key + "_sd"])]
    return out.to_string(index=False)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Decision-engine replay with component "
                                                 "ablation (evaluation plan D1).")
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP,
                        help="an eval_dump.py output folder (default: results/eval_dump/baseline)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="scaling rules (default: config.yaml; never config_demo.yaml "
                             "for report numbers)")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    started = time.time()
    with open(args.dump / "meta.json", encoding="utf-8") as f:
        dump_meta = json.load(f)
    settings = dump_meta["settings"]
    files = sorted(args.dump.glob("dump_seed*.csv"),
                   key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1)))
    if not files:
        raise FileNotFoundError(f"no dump_seed*.csv in {args.dump}; run model/eval_dump.py first")
    if settings["table_ii_scale_threshold"] != SPIKE_THRESHOLD:
        raise ValueError("the dump was scored with a different spike threshold")
    if _sha256(DATA_CSV) != dump_meta["data"]["sha256"]:
        raise ValueError("metrics.csv changed since the dump was made; re-run model/eval_dump.py")
    with open(args.config, encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}

    interval_s = settings["sampling_interval_seconds"]
    _, test_df = chronological_split(load_csv(DATA_CSV), settings["train_frac"])
    cpu = test_df["cpu_percent"].to_numpy(dtype=float)
    onsets = [i for i in range(1, len(cpu))
              if cpu[i] > SPIKE_THRESHOLD and cpu[i - 1] <= SPIKE_THRESHOLD]  # evaluate.py's rule
    if len(onsets) != settings["spike_onsets_in_test_split"]:
        raise ValueError(f"found {len(onsets)} spike onsets, the dump's meta says "
                         f"{settings['spike_onsets_in_test_split']}")
    print(f"[ProximaScale] D1 replay: {len(POLICIES)} policies x {len(files)} seed(s) from "
          f"{_display_path(args.dump)}, {len(onsets)} spike onsets, config "
          f"{_display_path(args.config)}", flush=True)

    per_seed, spike_rows, action_rows = [], [], []
    for path in files:
        dump = pd.read_csv(path, usecols=DUMP_COLUMNS)
        seed = int(dump["seed"].iloc[0])
        t_rows = dump["t"].to_numpy(dtype=int)
        if not np.all(np.diff(t_rows) == 1):
            raise ValueError(f"{path.name}: the replay needs one decision per row "
                             "(a dump made without --limit)")
        if not np.allclose(dump["current_cpu"].to_numpy(), cpu[t_rows]):
            raise ValueError(f"{path.name} is not aligned with metrics.csv's test split")
        seed_started = time.time()
        for policy in POLICIES:
            result = replay(policy, dump, base_config)
            metrics, spikes = score(result["signals"], result["reasons"], result["replicas"],
                                    t_rows, cpu, onsets, interval_s, settings["horizon"],
                                    base_config.get("max_containers", 10))
            per_seed.append({"policy": policy.name, "seed": seed, **metrics})
            spike_rows += [{"policy": policy.name, "seed": seed, **s,
                            "onset_ts": test_df["timestamp"].iloc[s["onset_row"]].isoformat()}
                           for s in spikes]
            action_rows += [{"policy": policy.name, "seed": seed, "t": int(t_rows[a["row"]]),
                             "ts": dump["ts_t"].iloc[a["row"]],
                             "actual_cpu": float(cpu[t_rows[a["row"]]]),
                             **{k: v for k, v in a.items() if k != "row"}}
                            for a in result["actions"]]
        print(f"[ProximaScale]   seed {seed}: {len(POLICIES)} policies replayed "
              f"({time.time() - seed_started:.0f}s)", flush=True)

    per_seed = pd.DataFrame(per_seed)
    grouped = per_seed.groupby("policy", sort=False)[list(METRICS)]
    summary = grouped.mean().join(grouped.std(ddof=1).add_suffix("_sd")).reset_index()
    summary.insert(1, "isolates", [p.isolates for p in POLICIES])
    summary.insert(2, "n_seeds", len(files))
    summary = summary[["policy", "isolates", "n_seeds"]
                      + [c for m in METRICS for c in (m, m + "_sd")]]

    out_dir = args.out_dir / args.dump.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.6g")
    per_seed.to_csv(out_dir / "per_seed.csv", index=False, float_format="%.6g")
    pd.DataFrame(spike_rows).to_csv(out_dir / "spikes.csv", index=False, float_format="%.6g")
    pd.DataFrame(action_rows).to_csv(out_dir / "actions.csv", index=False, float_format="%.6g")

    n_decisions = len(t_rows)
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": [Path(sys.argv[0]).name, *(sys.argv[1:] if argv is None else argv)],
        "runtime_seconds": round(time.time() - started, 1),
        "open_loop": "scaling actions do not change the replayed trace; this scores decision "
                     "quality on fixed data, not closed-loop dynamics",
        "dump": {"path": _display_path(args.dump),
                 "files": {p.name: _sha256(p) for p in files},
                 "git": dump_meta.get("git"), "settings": settings,
                 "artifacts": dump_meta.get("artifacts")},
        "config": {"path": _display_path(args.config), "values": base_config},
        "trace": {"decisions": n_decisions, "days": n_decisions * interval_s / 86400,
                  "decision_interval_seconds": interval_s, "spike_onsets": len(onsets),
                  "first_decision": str(dump["ts_t"].iloc[0]),
                  "last_decision": str(dump["ts_t"].iloc[-1]),
                  "initial_state": "replicas at min_containers, no cooldown armed, "
                                   "empty adaptive-threshold window"},
        "definitions": {"spike_onset": f"first row with actual CPU above {SPIKE_THRESHOLD:g}% "
                                       "after a row at or below it (evaluate.py)",
                        "spike_end": f"first row back at or below {SPIKE_THRESHOLD:g}%",
                        **METRICS},
        "policies": [asdict(p) for p in POLICIES],
        "git": _git_state(),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[ProximaScale] Decision-level results ({n_decisions} decisions = "
          f"{n_decisions * interval_s / 86400:.2f} days; mean +/- sd over {len(files)} seeds)",
          flush=True)
    print(_format_summary(summary, len(onsets)), flush=True)
    print(f"[ProximaScale] Done in {time.time() - started:.0f}s -> {_display_path(out_dir)}",
          flush=True)
    return summary, per_seed


if __name__ == "__main__":
    main()
