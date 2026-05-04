"""
main.py — ProximaScale orchestration loop.
Person D owns this file.

Modes:
  python main.py            → uses REAL LSTM model (requires TF + saved model)
  python main.py --simulate → simulation mode with hardcoded predictions (no GPU needed)

Pipeline:
  collect_metrics(window=10)   [Person A]
       ↓
  predict(records)             [Person B]
       ↓
  engine.evaluate(cpu, anomaly)[Person C]
       ↓
  execute(signal)              [Person D → DockerActuator]
"""
import os
import sys
import time
import logging
import argparse


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
    """
    Hardcoded predictions standing in for Person B's model.
    Use this when you don't want to load TensorFlow (e.g., quick demo).
    Replace with real predict() call for actual deployment.
    """
    return [
        (45.0, False),   # normal    → hold
        (80.0, False),   # high      → scale_up
        (82.0, False),   # cooldown  → hold_cooldown
        (88.0, True),    # anomaly   → scale_up (bypasses threshold)
        (25.0, False),   # low       → scale_down (after cooldown)
        (50.0, False),   # normal    → hold
    ]


# ── Real model loop ───────────────────────────────────────────────────────────
def run_real_loop(poll_interval: int = 60):
    """
    Production loop:
      1. Collect last 10 metric readings from monitoring CSV
      2. Call Person B's predict() to get CPU forecast + anomaly flag
      3. Evaluate with decision engine
      4. Execute scaling action
    """
    from monitoring.collector import collect_metrics
    from model.predict import predict            # Person B's interface

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

        result        = predict(records[-10:])   # always use the 10 most recent
        predicted_cpu = result["predicted_cpu"][0]   # use 1-step-ahead value for decision
        anomaly_flag  = result["anomaly"]

        logger.info(
            f"Predicted CPU (next 3 steps): {[round(v,1) for v in result['predicted_cpu']]} | "
            f"Anomaly: {anomaly_flag}"
        )

        raw_signal = engine.evaluate(predicted_cpu, anomaly_flag=anomaly_flag)
        signal     = normalise_signal(raw_signal)
        execute(signal)

        time.sleep(poll_interval)


# ── Simulation loop ───────────────────────────────────────────────────────────
def run_simulation():
    """Demo loop — no Docker or TensorFlow required."""
    from decision.engine import DecisionEngine
    from unittest.mock import MagicMock, patch

    logger.info("Simulation mode — using hardcoded predictions.")

    with patch("decision.engine.DockerActuator") as MockActuator:
        mock_actuator = MagicMock()
        mock_actuator.scale_up.return_value = True
        mock_actuator.scale_down.return_value = True
        MockActuator.return_value = mock_actuator

        sim_engine = DecisionEngine(config_path=CONFIG_PATH)

        for predicted_cpu, anomaly_flag in simulate_lstm_predictions():
            raw_signal = sim_engine.evaluate(predicted_cpu, anomaly_flag=anomaly_flag)
            logger.info(f"Raw signal: {raw_signal}")
            signal = normalise_signal(raw_signal)
            execute(signal)
            time.sleep(1)

    logger.info("Simulation complete.")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProximaScale orchestration loop")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    logger.info("ProximaScale starting...")

    if args.simulate:
        run_simulation()
    else:
        from decision.engine import DecisionEngine   # ← add here
        engine = DecisionEngine(config_path=CONFIG_PATH)
        try:
            run_real_loop(poll_interval=args.interval)
        except ImportError as e:
            logger.error(f"Could not load ML model: {e}")
            sys.exit(1)