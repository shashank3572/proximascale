import yaml
import time 
# Updated import based on Fix 4 (folder reorganization)
from actuator.docker_scaler import DockerActuator 

class DecisionEngine:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        self.upper_bound = self.config['scaling_rules']['cpu_upper_threshold']
        self.lower_bound = self.config['scaling_rules']['cpu_lower_threshold']
        self.last_action_time = 0
        self.cooldown_seconds = 180 
        
        self.actuator = DockerActuator(self.config)

    def evaluate(self, predicted_cpu, anomaly_flag=False):
        # FIX: Everything below must be indented inside the evaluate function
        print(f"\n📊 Engine: Received predicted CPU: {predicted_cpu}%, Anomaly: {anomaly_flag}")

        # 1. If anomaly detected, always scale up immediately
        if anomaly_flag:
            print("🚨 Engine: Anomaly detected! Forcing scale up.")
            action_taken = self.actuator.scale_up()
            if action_taken:
                self.last_action_time = time.time() # FIX: Reset the timer!
            return "scale_up" if action_taken else "hold_max_reached"

        # 2. Check cooldown — skip scaling if we acted too recently
        now = time.time()
        if (now - self.last_action_time) < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - (now - self.last_action_time))
            print(f"⏳ Engine: Cooldown active. {remaining}s remaining. Holding.")
            return "hold_cooldown"

        # 3. Threshold logic
        if predicted_cpu > self.upper_bound:
            action_taken = self.actuator.scale_up()
            if action_taken:
                self.last_action_time = time.time()  # FIX: Reset the timer!
            return "scale_up" if action_taken else "hold_max_reached"

        elif predicted_cpu < self.lower_bound:
            action_taken = self.actuator.scale_down()
            if action_taken:
                self.last_action_time = time.time()  # FIX: Reset the timer!
            return "scale_down" if action_taken else "hold_min_reached"

        else:
            print("⚖️ Engine: CPU is stable. Holding current state.")
            return "hold"