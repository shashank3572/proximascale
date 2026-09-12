# decision/engine.py
import time
from decision.hysteresis import Hysteresis
from decision.adaptive_threshold import AdaptiveThreshold
from decision.scaling_log import log_scaling_event


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
        Locked 3-arg interface:
            evaluate(predicted_cpu, upper_bound, anomaly_flag) -> str
        `upper_bound` is OPTIONAL. When None, the risk-aware branch is
        skipped, preserving Phase-1 2-arg call-site behaviour.

        Returns: "scale_up" | "scale_down" | "hold" | "hold_cooldown"
        """
        self.thresholds.update(predicted_cpu)
        upper_thresh, lower_thresh = self.thresholds.get_thresholds()

        # 1. Anomaly: bypasses cooldown AND risk branch
        if anomaly_flag:
            return self._do("scale_up", predicted_cpu, upper_bound,
                            reason="anomaly", upper_thresh=upper_thresh)

        # 2. Risk-aware: mean is fine but upper confidence bound is risky
        if upper_bound is not None and upper_bound > upper_thresh:
            return self._do("scale_up", predicted_cpu, upper_bound,
                            reason="upper_bound_risk", upper_thresh=upper_thresh)

        # 3. Cooldown gate for plain threshold-driven actions
        if self.hysteresis.is_cooling_down():
            return "hold_cooldown"

        # 4. Mean above upper threshold
        if predicted_cpu > upper_thresh:
            return self._do("scale_up", predicted_cpu, upper_bound,
                            reason="cpu_high", upper_thresh=upper_thresh)

        # 5. Mean below lower threshold
        if predicted_cpu < lower_thresh:
            return self._do("scale_down", predicted_cpu, upper_bound,
                            reason="cpu_low", lower_thresh=lower_thresh)

        # 6. Hold
        return "hold"

    # -------------------------------------------------------------------- _do
    def _do(self, action, predicted_cpu, upper_bound, reason, **extra):
        if self.actuator is not None:
            try:
                if action == "scale_up":
                    self.actuator.scale_up()
                elif action == "scale_down":
                    self.actuator.scale_down()
            except Exception:
                # Never let an actuator hiccup kill the decision loop
                pass

        self.hysteresis.record_action()

        log_scaling_event({
            "action": action,
            "timestamp": time.time(),
            "predicted_cpu": predicted_cpu,
            "upper_bound": upper_bound,
            "reason": reason,
            "expires_after_steps": self.cfg.get("cooldown_seconds", 180)
                                   // self.cfg.get("tick_seconds", 3),
            **extra,
        })
        return action