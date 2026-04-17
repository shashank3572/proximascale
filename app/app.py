from flask import Flask, request, jsonify
import sys
import os

# So Flask can find the model folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from predict import predict

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ProximaScale API is running!"})

@app.route('/predict', methods=['POST'])
def get_prediction():
    """
    Accepts 10 metric records and returns CPU predictions + anomaly flag.
    
    Expected input:
    {
        "records": [
            {
                "timestamp": "2024-01-15T14:32:00",
                "cpu_percent": 67.4,
                "memory_percent": 52.1,
                "request_rate": 143
            },
            ... (10 records total)
        ]
    }
    """
    data = request.get_json()

    # Validate input
    if not data or 'records' not in data:
        return jsonify({"error": "Missing records in request"}), 400

    if len(data['records']) != 10:
        return jsonify({"error": "Exactly 10 records required"}), 400

    # Run prediction
    result = predict(data['records'])

    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)