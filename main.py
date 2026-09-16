"""
main.py — ProximaScale orchestration loop.
Person D owns this file.

Modes:
  python main.py            → uses REAL LSTM model (requires TF + saved model)
  python main.py --simulate → simulation mode with hardcoded predictions (no GPU needed)

Pipeline:
  collect_metrics(window=10)                                [Person A]
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
import sys
import time
import logging
import argparse

from decision.scaling_log import clear_scaling_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


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
def run_real_loop(poll_interval: int = 30):
    """
    Production loop:
      1. Collect last 10 metric readings from monitoring CSV
      2. Call Person B's predict_load() to get CPU forecast + upper bound + anomaly flag
      3. Evaluate with decision engine (upper_bound-aware, Semester-2)
      4. Execute scaling action

    poll_interval defaults to 30s to match monitoring/collector.py's
    POLL_INTERVAL and the model's trained sampling rate (SAMPLING_INTERVAL_SEC
    in model/evaluate.py, DEFAULT_FREQ in model/prophet_model.py). Don't
    change one without the others -- lead-time and Prophet's forecast horizon
    are both computed assuming 30s between samples.
    """
    from monitoring.collector import collect_metrics
    from model.predict import predict_load        # Person B's Semester-2 interface

    logger.info(f"Real loop starting. Poll interval: {poll_interval}s")

    while True:
        records = collect_metrics(window=10)

        if len(records) < 10:
            logger.warning(
                f"Only {len(records)} records in CSV — need 10. "
                "Waiting for more data from collector..."
            )
            time.sleep(poll_interval)
            continue

        # Semester-2: predict_load() returns a 3-tuple, not the old
        # {"predicted_cpu": [...], "anomaly": bool} dict.
        predicted_cpu, upper_bound, anomaly_flag = predict_load(records[-10:])

        logger.info(
            f"Predicted CPU (~90s ahead): {round(predicted_cpu, 1)} | "
            f"Upper bound: {round(upper_bound, 1)} | Anomaly: {anomaly_flag}"
        )

        raw_signal = engine.evaluate(predicted_cpu, upper_bound, anomaly_flag=anomaly_flag)
        logger.info(f"→ {raw_signal} (reason={engine.last_reason})")

        signal = normalise_signal(raw_signal)
        execute(signal)

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

        sim_engine = DecisionEngine(config_path=CONFIG_PATH)

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
    args = parser.parse_args()

    logger.info("🚀 ProximaScale starting...")
    clear_scaling_events()

    if args.simulate:
        run_simulation()
    else:
        from decision.engine import DecisionEngine
        engine = DecisionEngine(config_path=CONFIG_PATH)
        try:
            run_real_loop(poll_interval=args.interval)
        except ImportError as e:
            logger.error(f"Could not load ML model: {e}")
            sys.exit(1)