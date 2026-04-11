import psutil
import time
import csv
from datetime import datetime

file_path = "data/collected/metrics.csv"

def collect_metrics():
    while True:
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        timestamp = datetime.now().isoformat()

        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, cpu, memory,0])

        print(f"{timestamp} | CPU: {cpu}% | Memory: {memory}%")

        time.sleep(5)

if __name__ == "__main__":
    collect_metrics()