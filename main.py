import os
import sys
import time
import logging
from decision.scaling_log import clear_scaling_events

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

    Each row: (predicted_cpu, upper_bound_or_None, anomaly_flag).
    None = 'Person B hasn't shipped MC-Dropout yet' → risk branch skipped.
    Swap for: from model.predict import get_prediction
    """
    return [
        (45.0, None, False),   # normal       → hold
        (80.0, None, False),   # high         → scale_up
        (82.0, None, False),   # high, in cd  → hold_cooldown
        (88.0, None, True),    # anomaly      → scale_up (bypasses cooldown)
        (25.0, None, False),   # low          → scale_down
        (50.0, 85.0, False),   # safe mean, risky bound → scale_up  ← NEW
        (50.0, None, False),   # normal       → hold
    ]


if __name__ == "__main__":
    clear_scaling_events()
    logger.info("🚀 ProximaScale Decision Engine starting...")

    predictions = simulate_lstm_predictions()

    for predicted_cpu, upper_bound, anomaly_flag in predictions:
    # Backward-compat: -inf means "no risk info" → branch skipped
        ub = upper_bound

        raw_signal = engine.evaluate(predicted_cpu, ub, anomaly_flag=anomaly_flag)
        logger.info(
            f"cpu={predicted_cpu} upper={ub} anomaly={anomaly_flag} "
            f"→ raw={raw_signal}"
        )

        signal = normalise_signal(raw_signal)
        execute(signal)

        time.sleep(2)
    logger.info("✅ Simulation complete.")
