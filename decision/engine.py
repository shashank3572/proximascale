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
        # Minimum interval before an anomaly scale-up may fire again, measured
        # from the last scale_up of ANY kind (seconds). The anomaly branch is
        # meant to beat the normal cooldown on a real emergency, but one
        # sustained spike used to add a worker on every poll (a live demo fired
        # two anomaly scale-ups 34 s apart), and an anomaly right after a
        # cpu_high scale-up added a second worker inside the same window.
        # 0 turns the gate off.
        self.anomaly_scale_up_min_interval = float(
            self.cfg.get("anomaly_scale_up_min_interval", 120)
        )
        self.last_scale_up_time = 0.0   # 0 = never acted, so never gated

        self.thresholds = AdaptiveThreshold(
            window_size=self.cfg.get("adaptive_window", 30),
            k=self.cfg.get("adaptive_k", 1.5),
            min_samples=self.cfg.get("adaptive_min_samples", 10),
            static_upper=self.cfg.get("cpu_upper_threshold", 75.0),
            static_lower=self.cfg.get("cpu_lower_threshold", 30.0),
            floor_lower=self.cfg.get("adaptive_floor_lower", 15.0),
            ceil_lower=self.cfg.get("adaptive_ceil_lower", 45.0),
            min_band=self.cfg.get("adaptive_min_band", 10.0),
        )

    # ---------------------------------------------------------------- evaluate
    def evaluate(self, predicted_cpu, upper_bound=None, anomaly_flag=False, current_cpu=None):
        """
         Returns one of "scale_up" | "scale_down" | "hold"  (locked 3-value contract).

            Why a non-scaling decision was made is exposed via `self.last_reason`,
            one of:
        "in_band"       — CPU inside the adaptive band
        "scale_down_guard" — the forecast says scale down, but the CURRENT
                          reading is still above scale_down_max_current_cpu
                          (only when current_cpu is passed and the key is set)
        "cooldown"      — still inside the hysteresis window
        "anomaly_min_interval" — an anomaly spike, but some scale_up was less
                          than anomaly_scale_up_min_interval ago (one spike,
                          one worker, whatever branch added the first one)
        "max_reached"   — scale_up requested, but actuator at max replicas
            "min_reached"   — scale_down requested, but actuator at min replicas
            "<reason>_no_actuator" — actuator unavailable (Docker down / not wired)
             "<reason>_error"       — SDK call raised
          "anomaly" | "upper_bound_risk" | "cpu_high" | "current_cpu_high" | "cpu_low"  — on success
        ("current_cpu_high": the reactive floor, scale_up_on_current_cpu)

        Ordering (and why the cooldown gate sits where it does):
          1. anomaly                — genuine emergencies bypass the normal
              cooldown, but only if no scale_up of any kind happened within
              anomaly_scale_up_min_interval seconds (one spike used to add a
              worker on every poll while the flag stayed set)
          2. cooldown               — gates upper_bound_risk too. The upper bound
              is a *statistical* risk signal, not a measured emergency; without
              this gate a decaying spike re-fires scale_up on every poll because
              the bound lags the mean downward (observed as the scale-up storm
              followed by scale_down/cooldown oscillation). One scale_up per
              cooldown window is the correct response to one spike.
          3. upper_bound_risk       — preemptive scale-up, cooldown-free only
          4. cpu_high               — measured mean above the upper threshold
          5. cpu_low                — mean below the lower threshold
          6. in_band                — nothing to do
         """
        self.last_reason = "in_band"

        self.thresholds.update(predicted_cpu)
        upper_thresh, lower_thresh = self.thresholds.get_thresholds()

    # 1. Anomaly bypasses the risk branch and cooldown, but it is still one
    #    scale_up per anomaly_scale_up_min_interval (see _anomaly_interval_active)
        if anomaly_flag and self.cfg.get("use_anomaly_bypass", True):
            if self._anomaly_interval_active():
                self.last_reason = "anomaly_min_interval"
                return "hold"
            return self._do(
                "scale_up",
                predicted_cpu,
                upper_bound,
                reason="anomaly",
                upper_thresh=upper_thresh,
            )

    # 2. Cooldown gates EVERY remaining branch (risk, high, low)
        if self.hysteresis.is_cooling_down():
            self.last_reason = "cooldown"
            return "hold"

    # 3. Risk-aware scale-up (mean safe, upper bound risky)
        if (upper_bound is not None and self.cfg.get("use_upper_bound_risk", True)
                and upper_bound > upper_thresh):
            return self._do(
                "scale_up",
                predicted_cpu,
                upper_bound,
                reason="upper_bound_risk",
                upper_thresh=upper_thresh,
            )

    # 4. Mean above upper bound
        if predicted_cpu > upper_thresh:
            return self._do(
                "scale_up",
                predicted_cpu,
                upper_bound,
                reason="cpu_high",
                upper_thresh=upper_thresh,
            )

    # 4b. Reactive floor: CPU is ALREADY above the threshold, whatever the
    #     forecast says, so scaling up is never later than a reactive scaler
    #     (a forecaster that misses a spike shape it was not trained on would
    #     otherwise leave the spike unhandled). Off unless set.
        if (self.cfg.get("scale_up_on_current_cpu", False) and current_cpu is not None
                and current_cpu > upper_thresh):
            return self._do(
                "scale_up",
                predicted_cpu,
                upper_bound,
                reason="current_cpu_high",
                upper_thresh=upper_thresh,
            )

    # 5. Mean below lower bound -- unless CPU is still high right now: a
    #    forecaster that sees a spike's end coming would otherwise remove
    #    capacity while the spike is still running.
        if predicted_cpu < lower_thresh:
            guard = self.cfg.get("scale_down_max_current_cpu")
            if guard is not None and current_cpu is not None and current_cpu > guard:
                self.last_reason = "scale_down_guard"
                return "hold"
            return self._do(
                "scale_down",
                predicted_cpu,
                upper_bound,
                reason="cpu_low",
                lower_thresh=lower_thresh,
         )

    # 6. Inside band
        return "hold"


    def _anomaly_interval_active(self):
        """True while the last scale_up of any kind is still inside the
        anomaly minimum interval, so one spike cannot add a second worker
        through the anomaly branch.

        Deliberately not Hysteresis: that clock is also armed by scale_down and
        is what the anomaly branch exists to bypass.
        """
        elapsed = time.time() - self.last_scale_up_time
        return elapsed < self.anomaly_scale_up_min_interval

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
        if action == "scale_up":
            self.last_scale_up_time = time.time()
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