"""
main.py — ProximaScale orchestration loop.
Person D owns this file.

Modes:
  python main.py            → uses REAL LSTM model (requires TF + saved model)
  python main.py --simulate → simulation mode with hardcoded predictions (no GPU needed)

Pipeline:
  monitoring/collector.py (separate process) -> metrics.csv  [Person A]
  storage.read_last_n(10)                                    [Person A]
       ↓
  predict_load(records) -> (cpu, upper_bound, anomaly)       [Person B, Semester-2]
       ↓
  engine.evaluate(cpu, upper_bound, anomaly_flag=anomaly)    [Person C, Semester-2]
       ↓
  execute(signal)                                            [Person D → DockerActuator]
       ↓
  ShapExplainer attribution on scale-up -> logs/events.csv   [Person D]

upper_bound (MC-Dropout mean + 2*std) is now passed straight into
DecisionEngine.evaluate(), which scales up preemptively if the upper bound
alone crosses the risk threshold -- see decision/engine.py's upper_bound
branch (Person C, Semester-2).
"""
import os
import csv
import sys
import time
import logging
import argparse
from datetime import datetime

from decision.scaling_log import clear_scaling_events

# ── Noise control ───────────────────────────────────────────────────────────
# TF, shap and joblib are all imported lazily at runtime, so setting these
# here is early enough. Without them a single decision cycle prints shap's
# subset-weight internals, absl/oneDNN C++ chatter, and a joblib core-probe
# traceback (WinError 2) — drowning the loop's own two lines per poll.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # absl/oneDNN C++ logs
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
# TF's Python-side "GPU not available on native Windows" notice
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# shap logs its subset-weight internals (num_full_subsets, phi arrays, ...)
# at INFO from its explainer modules
logging.getLogger("shap").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

WINDOW_SIZE = 10          # readings per prediction (model/predict.py WINDOW_SIZE)
STALE_AFTER_POLLS = 3     # skip prediction if newest CSV row is older than this many polls

# Dashboard feed (dashboard/live_plot.py reads this file; keep columns in sync).
EVENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "events.csv")
EVENT_COLUMNS = [
    "timestamp", "actual_cpu", "predicted_cpu", "upper_bound", "anomaly_flag",
    "signal", "replicas", "shap_cpu", "shap_memory", "shap_request", "reason",
]


# ── Signal normaliser ─────────────────────────────────────────────────────────
def normalise_signal(signal: str) -> str:
    """
    Collapses hold_cooldown / hold_max_reached / hold_min_reached
    into plain 'hold' so execute() never sees an unexpected variant.
    """
    return "hold" if signal.startswith("hold") else signal


def execute(signal: str):
    """Logging layer — DockerActuator already performed the action inside engine.
    Holds stay silent: the caller's '→ hold (reason=...)' line already says why."""
    if signal == "scale_up":
        logger.info("Action: scale_up executed.")
    elif signal == "scale_down":
        logger.info("Action: scale_down executed.")


# ── Simulation predictions (used with --simulate flag) ───────────────────────
def simulate_lstm_predictions():
    """Dummy predictions standing in for Person B's model output.

    Each row: (predicted_cpu, upper_bound_or_None, anomaly_flag).
    None = 'Person B hasn't shipped MC-Dropout yet' -> risk branch skipped.
    Swap for: from model.predict import predict_load
    """
    return [
        (45.0, None, False),   # normal                 → hold
        (80.0, None, False),   # high                    → scale_up
        (82.0, None, False),   # high, in cooldown       → hold_cooldown
        (88.0, None, True),    # anomaly                 → scale_up (bypasses cooldown)
        (25.0, None, False),   # low                     → scale_down
        (50.0, 85.0, False),   # safe mean, risky bound  → scale_up
        (50.0, None, False),   # normal                  → hold
    ]


# ── Real model loop ───────────────────────────────────────────────────────────
def log_event(row: dict, path: str = None) -> None:
    """Append one decision row to logs/events.csv for the dashboard.
    Never raises: a logging failure must not kill the control loop.

    If an existing file was written with an older column set (e.g. before the
    `reason` column existed), it is migrated in place first -- mixed-width
    rows would otherwise crash the dashboard's read_csv.
    """
    path = path or EVENTS_PATH
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        if not new_file:
            with open(path, "r", newline="") as f:
                first_line = f.readline()
            existing_cols = next(csv.reader([first_line]), [])
            if existing_cols and existing_cols != EVENT_COLUMNS:
                _migrate_events_file(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow({c: row.get(c, 0.0) for c in EVENT_COLUMNS})
    except OSError as e:
        logger.warning(f"Could not write dashboard event log: {e}")


def _migrate_events_file(path: str) -> None:
    """Rewrite an events.csv that used an older column layout into the
    current EVENT_COLUMNS layout (missing values become 0.0 / empty).
    Raises OSError to the caller, which treats it as non-fatal."""
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        old_rows = list(reader)
        old_fields = reader.fieldnames or []

    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        for old in old_rows:
            migrated = {}
            for c in EVENT_COLUMNS:
                value = old.get(c)
                if c == "reason":
                    migrated[c] = value if value not in (None, "") else "pre-reason log"
                else:
                    migrated[c] = value if value not in (None, "") else 0.0
            writer.writerow(migrated)
    os.replace(tmp, path)
    logger.info(f"Migrated {path} from {len(old_fields)} columns to "
                f"{len(EVENT_COLUMNS)} ({len(old_rows)} rows kept).")


def _replica_count(engine) -> int:
    """Current worker count, or 0 if the actuator is unavailable."""
    try:
        return int(engine.actuator.replica_count())
    except Exception:
        return 0


def _is_stale(record, max_age_seconds: float) -> bool:
    """True if the newest collected record is too old (collector not running)
    or its timestamp cannot be parsed."""
    try:
        age = (datetime.now() - datetime.fromisoformat(record.timestamp)).total_seconds()
    except (ValueError, TypeError):
        return True
    return age > max_age_seconds


def _compute_shap(window_records):
    """SHAP attribution for the current window, or None if it can't run.

    Uses prepare_scaled_window() so the explained input is EXACTLY the input
    predict_load() fed to the LSTM, and the model/cache loaded during
    prediction -- no duplicate .h5 load, no scaling drift. Never raises: a
    SHAP failure must not kill the control loop, the dashboard just shows
    'no attribution' for that event.
    """
    try:
        import numpy as np
        import pandas as pd
        from model.predict import prepare_scaled_window, get_shap_artifacts
        from dashboard.shap_explain import ShapExplainer

        # Accept MetricRecord objects from the monitoring loop.
        window_records = [
            r.to_dict() if hasattr(r, "to_dict") else r for r in window_records
        ]
        scaled, _ = prepare_scaled_window(window_records)
        model, _feature_scaler = get_shap_artifacts()

        # Background = the current window itself, PLUS a neutral flat-50%
        # reference window scaled through the real feature scaler. kmeans on
        # the two points keeps KernelExplainer cheap (nsamples=256) while
        # giving it low/high baselines to attribute against -- with only one
        # background point equal to the input, every attribution would
        # degenerate to ~0.
        neutral = pd.DataFrame({
            "timestamp": [r["timestamp"] for r in window_records],
            "cpu_percent": [50.0] * len(window_records),
            "memory_percent": [50.0] * len(window_records),
            "request_rate": [0] * len(window_records),
            "post_scaling": [False] * len(window_records),
        })
        neutral_scaled, _ = prepare_scaled_window(neutral.to_dict("records"))
        background = np.stack([scaled, neutral_scaled]).astype(np.float32)

        explainer = ShapExplainer(model, background, nsamples=256)
        return explainer.explain(scaled.reshape(1, 10, 3).astype(np.float32))
    except Exception as e:
        logger.warning(f"SHAP attribution unavailable this cycle: {e}")
        return None


def run_real_loop(engine, poll_interval: int = 30):
    """
    Production loop:
      1. Read the last 10 readings from metrics.csv (written by the separate
         `python -m monitoring.collector` process)
      2. Call Person B's predict_load() to get CPU forecast + upper bound + anomaly flag
      3. Evaluate with decision engine (upper_bound-aware, Semester-2)
      4. Execute scaling action
      5. Log the decision + (on scale-up) SHAP attribution for the dashboard

    The loop never scales on stale data: if the newest row is older than
    STALE_AFTER_POLLS * poll_interval it warns and waits.

    poll_interval defaults to 30s to match monitoring/collector.py's
    POLL_INTERVAL and the model's trained sampling rate (SAMPLING_INTERVAL_SEC
    in model/evaluate.py, DEFAULT_FREQ in model/prophet_model.py). Don't
    change one without the others -- lead-time and Prophet's forecast horizon
    are both computed assuming 30s between samples.
    """
    from monitoring.storage import read_last_n
    from model.predict import predict_load        # Person B's Semester-2 interface

    logger.info(f"Real loop starting. Poll interval: {poll_interval}s")

    while True:
        records = read_last_n(WINDOW_SIZE)

        if len(records) < WINDOW_SIZE:
            logger.warning(
                f"Only {len(records)} records in CSV — need {WINDOW_SIZE}. "
                "Waiting for more data from collector..."
            )
            time.sleep(poll_interval)
            continue

        if _is_stale(records[-1], STALE_AFTER_POLLS * poll_interval):
            logger.warning(
                f"Newest record ({records[-1].timestamp}) is stale — "
                "is `python -m monitoring.collector` running? Skipping this cycle."
            )
            time.sleep(poll_interval)
            continue

        # predict_load() accepts MetricRecord objects directly and returns
        # (predicted_cpu, upper_bound, anomaly_flag).
        predicted_cpu, upper_bound, anomaly_flag = predict_load(records)

        raw_signal = engine.evaluate(predicted_cpu, upper_bound, anomaly_flag=anomaly_flag)
        reason = engine.last_reason
        signal = normalise_signal(raw_signal)

        # One line per poll — the whole decision at a glance. Real scale
        # actions add an 'Action:' line below; holds stay at exactly this line.
        logger.info(
            f"pred {predicted_cpu:.1f} | ub {upper_bound:.1f} | "
            f"actual {records[-1].cpu_percent:.1f} | anomaly={anomaly_flag} "
            f"→ {signal} ({reason})"
        )
        execute(signal)

        # SHAP attribution only on scale-up events: KernelExplainer costs
        # ~256 model evaluations per call, which is wasteful on holds. On
        # failure it returns None and the row simply carries zeros.
        shap_row = {"shap_cpu": 0.0, "shap_memory": 0.0, "shap_request": 0.0}
        if signal == "scale_up":
            attribution = _compute_shap(records)
            if attribution is not None:
                logger.info(
                    "SHAP: " + ", ".join(f"{k} {v:.1f}%" for k, v in attribution.items())
                )
                shap_row = {
                    "shap_cpu": attribution.get("CPU%", 0.0),
                    "shap_memory": attribution.get("Memory%", 0.0),
                    "shap_request": attribution.get("Request Rate", 0.0),
                }

        log_event({
            "timestamp": datetime.now().isoformat(),
            "actual_cpu": records[-1].cpu_percent,
            "predicted_cpu": predicted_cpu,
            "upper_bound": upper_bound,
            "anomaly_flag": anomaly_flag,
            "signal": signal,
            "replicas": _replica_count(engine),
            **shap_row,
            "reason": engine.last_reason,
        })

        time.sleep(poll_interval)


# ── Simulation loop ───────────────────────────────────────────────────────────
def run_simulation():
    """Demo loop — no Docker or TensorFlow required."""
    from decision.engine import DecisionEngine
    from unittest.mock import MagicMock, patch

    logger.info("Simulation mode — using hardcoded predictions.")

    with patch("actuator.docker_scaler.DockerActuator") as MockActuator:
        mock_actuator = MagicMock()
        mock_actuator.scale_up.return_value = True
        mock_actuator.scale_down.return_value = True
        MockActuator.return_value = mock_actuator

        sim_engine = DecisionEngine(config_path=DEFAULT_CONFIG_PATH)

        for predicted_cpu, upper_bound, anomaly_flag in simulate_lstm_predictions():
            raw_signal = sim_engine.evaluate(predicted_cpu, upper_bound,
                                              anomaly_flag=anomaly_flag)
            logger.info(
                f"cpu={predicted_cpu} upper={upper_bound} anomaly={anomaly_flag} "
                f"→ {raw_signal} (reason={sim_engine.last_reason})"
            )
            signal = normalise_signal(raw_signal)
            execute(signal)
            time.sleep(1)

    logger.info("Simulation complete.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProximaScale orchestration loop")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH,
        help="Path to a scaling-rules config (default: config.yaml). "
             "Use config_demo.yaml for a faster-converging scale_down during "
             "a live demo -- see that file's header for why the real "
             "config.yaml needs ~15 min of idle time to show one. Never use "
             "config_demo.yaml when generating numbers for the report.",
    )
    args = parser.parse_args()
    os.environ["PROXIMASCALE_CONFIG_PATH"] = args.config

    logger.info("🚀 ProximaScale starting...")
    if args.config != DEFAULT_CONFIG_PATH:
        logger.warning(f"Using non-default config: {args.config} (demo settings, not for report numbers)")
    clear_scaling_events()

    if args.simulate:
        run_simulation()
    else:
        from decision.engine import DecisionEngine
        engine = DecisionEngine(config_path=args.config)
        try:
            run_real_loop(engine, poll_interval=args.interval)
        except KeyboardInterrupt:
            logger.info("Stopped by user (Ctrl+C) — events saved in logs/events.csv.")
        except ImportError as e:
            logger.error(f"Could not load ML model: {e}")
            sys.exit(1)