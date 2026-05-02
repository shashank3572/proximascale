import yaml
from actuator.docker_scaler import DockerActuator
from decision.hysteresis import Hysteresis


class DecisionEngine:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.upper_bound = self.config['scaling_rules']['cpu_upper_threshold']
        self.lower_bound = self.config['scaling_rules']['cpu_lower_threshold']

        # Hysteresis now lives in its own class (extracted from inline code)
        self.hysteresis = Hysteresis(cooldown_seconds=180)

        self.actuator = DockerActuator(self.config)

    def evaluate(self, predicted_cpu, anomaly_flag=False):
        print(f"\n📊 Engine: Received predicted CPU: {predicted_cpu}%, Anomaly: {anomaly_flag}")

        # 1. Anomaly bypass — always scale up immediately, skip cooldown
        if anomaly_flag:
            print("🚨 Engine: Anomaly detected! Forcing scale up.")
            action_taken = self.actuator.scale_up()
            if action_taken:
                self.hysteresis.record_action()
            return "scale_up" if action_taken else "hold_max_reached"

        # 2. Cooldown check — don't act if we acted too recently
        if self.hysteresis.is_cooling_down():
            remaining = self.hysteresis.seconds_remaining()
            print(f"⏳ Engine: Cooldown active. {remaining}s remaining. Holding.")
            return "hold_cooldown"

        # 3. Threshold logic
        if predicted_cpu > self.upper_bound:
            action_taken = self.actuator.scale_up()
            if action_taken:
                self.hysteresis.record_action()
            return "scale_up" if action_taken else "hold_max_reached"

        elif predicted_cpu < self.lower_bound:
            action_taken = self.actuator.scale_down()
            if action_taken:
                self.hysteresis.record_action()
            return "scale_down" if action_taken else "hold_min_reached"

        else:
            print("⚖️ Engine: CPU is stable. Holding current state.")
            return "hold"
