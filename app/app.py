from flask import Flask
import time
import math
request_count=0

app = Flask(__name__)
@app.before_request
def count_request():
    global request_count
    request_count += 1

@app.route("/")
def home():
    result = 0
    for i in range(1000000):
        result += math.sqrt(i)
    return "App is running"

@app.route("/heavy")
def heavy():
    time.sleep(2)
    return "Heavy load"

import threading

def reset_counter():
    global request_count
    while True:
        time.sleep(60)
        print("Requests per minute:", request_count)
        request_count = 0

threading.Thread(target=reset_counter, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=True)