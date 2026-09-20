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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

WINDOW_SIZE = 10          # readings per prediction (model/predict.py WINDOW_SIZE)
STALE_AFTER_POLLS = 3     # skip prediction if newest CSV row is older than this many polls

# Dashboard feed (dashboard/live_plot.py reads this file; keep columns in sync).
EVENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "events.csv")
EVENT_COLUMNS = [
    "timestamp", "actual_cpu", "predicted_cpu", "upper_bound", "anomaly_flag",
    "signal", "replicas", "shap_cpu", "shap_memory", "shap_request",
]


# ── Signal normaliser ─────────────────────────────────────────────────────────
def normalise_signal(signal: str) -> str:
    """
    Collapses hold_cooldown / hold_max_reached / hold_min_reached
    into plain 'hold' so execute() never sees an unexpected variant.
    """
    return "hold" if signal.startswith("hold") else signal


def execute(signal: str):
    """Logging layer — DockerActuator already performed the action inside engine."""
    if signal == "scale_up":
        logger.info("Action: scale_up executed.")
    elif signal == "scale_down":
        logger.info("Action: scale_down executed.")
    elif signal == "hold":
        logger.info("Action: hold — no change.")


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
    Never raises: a logging failure must not kill the control loop."""
    path = path or EVENTS_PATH
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow({c: row.get(c, 0.0) for c in EVENT_COLUMNS})
    except OSError as e:
        logger.warning(f"Could not write dashboard event log: {e}")


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


def run_real_loop(engine, poll_interval: int = 30):
    """
    Production loop:
      1. Read the last 10 readings from metrics.csv (written by the separate
         `python -m monitoring.collector` process)
      2. Call Person B's predict_load() to get CPU forecast + upper bound + anomaly flag
      3. Evaluate with decision engine (upper_bound-aware, Semester-2)
      4. Execute scaling action

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

        logger.info(
            f"Predicted CPU (~90s ahead): {round(predicted_cpu, 1)} | "
            f"Upper bound: {round(upper_bound, 1)} | Anomaly: {anomaly_flag}"
        )

        raw_signal = engine.evaluate(predicted_cpu, upper_bound, anomaly_flag=anomaly_flag)
        logger.info(f"→ {raw_signal} (reason={engine.last_reason})")

        signal = normalise_signal(raw_signal)
        execute(signal)

        log_event({
            "timestamp": datetime.now().isoformat(),
            "actual_cpu": records[-1].cpu_percent,
            "predicted_cpu": predicted_cpu,
            "upper_bound": upper_bound,
            "anomaly_flag": anomaly_flag,
            "signal": signal,
            "replicas": _replica_count(engine),
            # SHAP attribution is not wired into the live loop yet; the
            # dashboard shows "No SHAP values recorded" for zero rows.
            "shap_cpu": 0.0, "shap_memory": 0.0, "shap_request": 0.0,
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
        except ImportError as e:
            logger.error(f"Could not load ML model: {e}")
            sys.exit(1)