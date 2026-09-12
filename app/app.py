from flask import Flask
import math
from monitoring.metrics import increment_request_count

app = Flask(__name__)


@app.before_request
def count_request():
    increment_request_count()


@app.route("/work/light")
def light_work():
    result = 0
    for i in range(200000):
        result += math.sqrt(i)
    return "Light work completed"


@app.route("/work/heavy")
def heavy_work():
    result = 0
    for i in range(5000000):
        result += math.sqrt(i)
    return "Heavy work completed"


@app.route("/health")
def health():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)