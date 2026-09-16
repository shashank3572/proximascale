# decision/engine.py
import time
from decision.hysteresis import Hysteresis
from decision.adaptive_threshold import AdaptiveThreshold
from decision.scaling_log import log_scaling_event
import logging
logger = logging.getLogger(__name__)

class DecisionEngine:
    def __init__(self, config=None, actuator=None, config_path=None):
        # --- Accept either a dict OR a path (backward compat) ---
        if config is None and config_path is not None:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        if config is None:
            config = {}

        # --- Flatten nested scaling_rules so both config styles work ---
        flat = dict(config)
        flat.update(config.get("scaling_rules", {}))
        self.cfg = flat

        # --- Auto-construct actuator unless one was passed in ---
        # This is what the existing test fixture relies on
        # (engine.actuator.client.containers.list.return_value = [...])
        self.actuator = actuator
        if self.actuator is None:
            try:
                from actuator.docker_scaler import DockerActuator
                self.actuator = DockerActuator(self.cfg)
            except Exception:
                # Docker SDK missing / daemon down → engine still works,
                # just no real side effects. Tests that mock docker.from_env
                # will succeed here.
                self.actuator = None

        self.hysteresis = Hysteresis(
            cooldown_seconds=self.cfg.get("cooldown_seconds", 180)
        )
        self.thresholds = AdaptiveThreshold(
            window_size=self.cfg.get("adaptive_window", 30),
            k=self.cfg.get("adaptive_k", 1.5),
            min_samples=self.cfg.get("adaptive_min_samples", 10),
            static_upper=self.cfg.get("cpu_upper_threshold", 75.0),
            static_lower=self.cfg.get("cpu_lower_threshold", 30.0),
            min_band=self.cfg.get("adaptive_min_band", 10.0),
        )

    # ---------------------------------------------------------------- evaluate
    def evaluate(self, predicted_cpu, upper_bound=None, anomaly_flag=False):
        """
         Returns one of "scale_up" | "scale_down" | "hold"  (locked 3-value contract).

            Why a non-scaling decision was made is exposed via `self.last_reason`,
            one of:
        "in_band"       — CPU inside the adaptive band
        "cooldown"      — still inside the hysteresis window
        "max_reached"   — scale_up requested, but actuator at max replicas
            "min_reached"   — scale_down requested, but actuator at min replicas
            "<reason>_no_actuator" — actuator unavailable (Docker down / not wired)
             "<reason>_error"       — SDK call raised
          "anomaly" | "upper_bound_risk" | "cpu_high" | "cpu_low"  — on success
         """
        self.last_reason = "in_band"

        self.thresholds.update(predicted_cpu)
        upper_thresh, lower_thresh = self.thresholds.get_thresholds()

    # 1. Anomaly bypasses both risk branch and cooldown
        if anomaly_flag:
            return self._do(
                "scale_up",
                predicted_cpu,
                upper_bound,
                reason="anomaly",
                upper_thresh=upper_thresh,
            )

    # 2. Risk-aware scale-up (mean safe, upper bound risky)
        if upper_bound is not None and upper_bound > upper_thresh:
            return self._do(
                "scale_up",
                predicted_cpu,
                upper_bound,
                reason="upper_bound_risk",
                upper_thresh=upper_thresh,
            )

    # 3. Cooldown gates only plain threshold-driven actions
        if self.hysteresis.is_cooling_down():
            self.last_reason = "cooldown"
            return "hold"

    # 4. Mean above upper bound
        if predicted_cpu > upper_thresh:
            return self._do(
                "scale_up",
                predicted_cpu,
                upper_bound,
                reason="cpu_high",
                upper_thresh=upper_thresh,
            )

    # 5. Mean below lower bound
        if predicted_cpu < lower_thresh:
            return self._do(
                "scale_down",
                predicted_cpu,
                upper_bound,
                reason="cpu_low",
                lower_thresh=lower_thresh,
         )

    # 6. Inside band
        return "hold"


    def _do(self, action, predicted_cpu, upper_bound, reason, **extra):
        """
         Execute one actuator action. Log + arm cooldown ONLY on actual success.
         Returns "scale_up"/"scale_down" on real scaling, "hold" otherwise.
        """
        if self.actuator is None:
            self.last_reason = f"{reason}_no_actuator"
            return "hold"

        try:
            if action == "scale_up":
                ok = self.actuator.scale_up()
            elif action == "scale_down":
                ok = self.actuator.scale_down()
            else:
                ok = False
        except Exception:
            logger.exception("Actuator %s raised — treating as hold", action)
            self.last_reason = f"{reason}_error"
            return "hold"

        if not ok:
        # Capacity guard tripped inside the actuator
            self.last_reason = (
                "max_reached" if action == "scale_up" else "min_reached"
            )
            return "hold"

    # -------- success path only --------
        self.hysteresis.record_action()
        self.last_reason = reason

        log_scaling_event({
            "action": action,
            "timestamp": time.time(),
            "predicted_cpu": predicted_cpu,
            "upper_bound": upper_bound,
            "reason": reason,
            "expires_after_steps": (
                self.cfg.get("cooldown_seconds", 180)
                // self.cfg.get("tick_seconds", 3)
            ),
            **extra,
        })

        return action