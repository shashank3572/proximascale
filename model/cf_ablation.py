"""
ProximaScale - Evaluation D2: counterfactual correction off at inference
------------------------------------------------------------------------
Compares two eval dumps made from the same models and MC seeds, one with the
correction on (results/eval_dump/baseline) and one with every post_scaling
flag cleared (eval_dump.py --no-counterfactual, tag cf_off). The correction
only touches windows containing post-scaling rows, so the comparison is
reported there, against both the recorded CPU and the generator's true load
(the CPU before its 0.8x post-scaling drop). No TensorFlow needed.

    python model/cf_ablation.py
    python model/cf_ablation.py --on results/eval_dump/baseline --off results/eval_dump/cf_off

Writes results/forecast_metrics/cf_ablation.csv.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DUMPS = REPO_ROOT / "results" / "eval_dump"
DEFAULT_OUT = REPO_ROOT / "results" / "forecast_metrics" / "cf_ablation.csv"

METHODS = {"predicted_load": "ProximaScale (hybrid mean)", "univariate_t3": "Univariate LSTM"}
TARGETS = {"actual_t3": "recorded", "true_cpu_t3": "true load"}
COLUMNS = ["seed", "t", "actual_t3", "true_cpu_t3", "regime_t3", "post_scaling_in_window",
           "anomaly_flag", *METHODS]


def load(dump_dir):
    meta = json.loads((dump_dir / "meta.json").read_text())
    files = sorted(dump_dir.glob("dump_seed*.csv"))
    if not files:
        raise FileNotFoundError(f"no dump_seed*.csv in {dump_dir}")
    return meta, {int(f.stem.removeprefix("dump_seed")): pd.read_csv(f, usecols=COLUMNS)
                  for f in files}


def groups(ref):
    in_window = ref["post_scaling_in_window"].to_numpy()
    target_post = (ref["regime_t3"] == "post_scaling").to_numpy()
    return {"all": np.ones(len(ref), dtype=bool),
            "post-scaling in window": in_window,
            "target is post-scaling": target_post,
            "post-scaling in window, target not": in_window & ~target_post}


def compare(on, off):
    """One row per (rows, method, target): per-seed RMSE and bias with the
    correction on and off, and the paired difference (on - off)."""
    seeds = sorted(set(on) & set(off))
    if not seeds:
        raise ValueError("the two dumps share no seeds")
    ref = on[seeds[0]]
    for seed in seeds:
        for col in ("t", "actual_t3", "true_cpu_t3", "regime_t3"):
            if not on[seed][col].equals(off[seed][col]):
                raise ValueError(f"seed {seed}: dumps differ in {col}; not the same windows")
    rows = []
    for name, mask in groups(ref).items():
        for method, label in METHODS.items():
            for target, target_label in TARGETS.items():
                def stats(d):
                    err = d[method].to_numpy()[mask] - d[target].to_numpy()[mask]
                    return np.sqrt(np.mean(err ** 2)), err.mean()
                s_on = np.array([stats(on[s]) for s in seeds])
                s_off = np.array([stats(off[s]) for s in seeds])
                diff = s_on[:, 0] - s_off[:, 0]
                rows.append({
                    "rows": name, "n": int(mask.sum()), "method": label, "vs": target_label,
                    "rmse_on": s_on[:, 0].mean(), "rmse_off": s_off[:, 0].mean(),
                    "d_rmse": diff.mean(), "d_rmse_sd": diff.std(ddof=1) if len(seeds) > 1 else 0.0,
                    "bias_on": s_on[:, 1].mean(), "bias_off": s_off[:, 1].mean()})
    flips = sum(int((on[s]["anomaly_flag"] != off[s]["anomaly_flag"]).sum()) for s in seeds)
    return pd.DataFrame(rows), seeds, flips


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluation D2: counterfactual off.")
    parser.add_argument("--on", type=Path, default=DUMPS / "baseline")
    parser.add_argument("--off", type=Path, default=DUMPS / "cf_off")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    meta_on, on = load(args.on)
    meta_off, off = load(args.off)
    if meta_on["settings"]["counterfactual_at_inference"] is not True or \
            meta_off["settings"]["counterfactual_at_inference"] is not False:
        raise ValueError("--on must be a correction-on dump and --off a --no-counterfactual one")
    if meta_on["artifacts"] != meta_off["artifacts"]:
        raise ValueError("the two dumps were made from different model artifacts")

    table, seeds, flips = compare(on, off)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.4f")

    print(f"[ProximaScale] D2: correction on minus off, {len(seeds)} seeds "
          f"({args.on.name} vs {args.off.name})")
    with pd.option_context("display.width", 200):
        print(table.round(3).to_string(index=False))
    print(f"  anomaly flags that change: {flips} over {len(seeds)} seeds")
    print(f"[ProximaScale] Done -> {args.out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
