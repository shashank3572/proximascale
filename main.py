import os
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Fix 1: Absolute config path so this works from any working directory ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

# --- Fix 2: Guard DockerActuator instantiation ---
# DockerActuator.__init__ calls client.images.pull() immediately.
# If Docker Desktop is not running this would crash before anything else runs.
try:
    from decision.engine import DecisionEngine
    engine = DecisionEngine(config_path=CONFIG_PATH)
    logger.info("DecisionEngine initialised successfully.")
except Exception as e:
    logger.error(f"Startup failed — Docker may not be running: {e}")
    sys.exit(1)


def normalise_signal(signal: str) -> str:
    """
    Fix 3: Collapse all 'hold_*' variants into plain 'hold' before
    passing to execute(). Prevents crashes on hold_cooldown /
    hold_max_reached / hold_min_reached.
    """
    return "hold" if signal.startswith("hold") else signal


def execute(signal: str):
    """Person D's actuator dispatch. engine.evaluate() already ran the
    Docker action, so this layer is just for logging / future hooks."""
    if signal == "scale_up":
        logger.info("✅ Action: scale_up executed.")
    elif signal == "scale_down":
        logger.info("✅ Action: scale_down executed.")
    elif signal == "hold":
        logger.info("➡️  Action: hold — no change.")


def simulate_lstm_predictions():
    """Dummy predictions standing in for Person B's model output.
    Swap this for: from model.predict import get_prediction"""
    return [
        (45.0, False),   # normal → hold
        (80.0, False),   # high   → scale_up
        (82.0, False),   # still high but cooldown active → hold_cooldown
        (88.0, True),    # anomaly → scale_up regardless
        (25.0, False),   # low    → scale_down (after cooldown)
        (50.0, False),   # normal → hold
    ]


if __name__ == "__main__":
    logger.info("🚀 ProximaScale Decision Engine starting...")

    predictions = simulate_lstm_predictions()

    for predicted_cpu, anomaly_flag in predictions:
        raw_signal = engine.evaluate(predicted_cpu, anomaly_flag=anomaly_flag)
        logger.info(f"Raw signal from engine: {raw_signal}")

        signal = normalise_signal(raw_signal)   # Fix 3 applied here
        execute(signal)

        time.sleep(2)   # Pause between cycles (use 30–60s in real deployment)

    logger.info("✅ Simulation complete.")
