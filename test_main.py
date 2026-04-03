import time
from decision_engine import DecisionEngine

def simulate_lstm_predictions():
    # A fake wave of traffic: Starts normal, spikes high, then drops
    return [45.0, 78.5, 82.0, 88.0, 60.0, 25.0, 15.0, 50.0]

if __name__ == "__main__":
    print("🚀 Starting AI SysAdmin Dummy Test...")
    engine = DecisionEngine()
    
    # Let's ensure we start with at least 1 container
    if len(engine.actuator.get_workers()) == 0:
        engine.actuator.scale_up()
    
    predictions = simulate_lstm_predictions()
    
    for cpu in predictions:
        # This is the exact function Person D will call later
        result = engine.evaluate(cpu)
        print(f"➡️ Interface Output: {result}")
        time.sleep(3) # Pause so you can watch Docker Desktop react!