import yaml
from actuator import DockerActuator

class DecisionEngine:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        self.upper_bound = self.config['scaling_rules']['cpu_upper_threshold']
        self.lower_bound = self.config['scaling_rules']['cpu_lower_threshold']
        
        self.actuator = DockerActuator(self.config)

    def evaluate(self, predicted_cpu):
        print(f"\n📊 Engine: Received predicted CPU: {predicted_cpu}%")
        
        if predicted_cpu > self.upper_bound:
            action_taken = self.actuator.scale_up()
            return "scale_up" if action_taken else "hold_max_reached"
            
        elif predicted_cpu < self.lower_bound:
            action_taken = self.actuator.scale_down()
            return "scale_down" if action_taken else "hold_min_reached"
            
        else:
            # Hysteresis: If it's between 30 and 75, we do nothing.
            print("⚖️ Engine: CPU is stable. Holding current state.")
            return "hold"