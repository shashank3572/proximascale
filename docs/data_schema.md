# ProximaScale — Data Schema Contract

> **This document is the team contract. Nobody changes the schema without a team discussion.**
> Written by Person D (Week 2). Referenced by Person A (produces), Person B (consumes).

---

## Metric Record — JSON / Dict Format

```json
{
  "timestamp":      "2024-01-15T14:32:00",
  "cpu_percent":    67.4,
  "memory_percent": 52.1,
  "request_rate":   143
}
```

| Field            | Type    | Unit / Range     | Notes                                  |
|------------------|---------|------------------|----------------------------------------|
| `timestamp`      | string  | ISO-8601         | `YYYY-MM-DDTHH:MM:SS`, no timezone     |
| `cpu_percent`    | float   | 0.0 – 100.0      | System-wide CPU utilisation            |
| `memory_percent` | float   | 0.0 – 100.0      | RAM utilisation (psutil virtual_memory)|
| `request_rate`   | integer | 0 – ∞            | HTTP requests received in last 60 s    |

---

## CSV File Format

File: `data/collected/metrics.csv`

```
timestamp,cpu_percent,memory_percent,request_rate
2024-01-15T14:32:00,67.4,52.1,143
2024-01-15T14:33:00,72.1,54.3,201
```

- Header row always present (written by `monitoring/storage.py`)
- One row per polling interval (default: every 10 seconds)
- Parsed by `monitoring/schema.py → MetricRecord.from_csv_row()`

---

## Python Class

```python
from monitoring.schema import MetricRecord

# From dict (JSON payload)
record = MetricRecord.from_dict(d)

# From CSV row (list of strings)
record = MetricRecord.from_csv_row(row)

# To dict (hand off to Person B)
d = record.to_dict()
```

---

## Interface Between Modules

```
Person A (monitoring/collector.py)
    produces → MetricRecord objects → stored as CSV

Person B (model/predict.py)
    consumes → list of 10 MetricRecord dicts (last 10 readings)
    call:   predict(records)   # records = [dict, dict, ... x10]
    returns:
        {
          "predicted_cpu": [float, float, float],  # next 3 timesteps
          "anomaly":       bool                    # True = spike detected
        }
Person C (decision/engine.py)
    consumes → predicted_cpu (float), anomaly (bool)
    call:   engine.evaluate(predicted_cpu, anomaly_flag)
    returns: "scale_up" | "scale_down" | "hold" | "hold_cooldown"
             | "hold_max_reached" | "hold_min_reached"

    Return value meanings:
        scale_up         — CPU above upper threshold, container added
        scale_down       — CPU below lower threshold, container removed
        hold             — CPU in stable range, no action
        hold_cooldown    — action blocked, cooldown period still active (180s)
        hold_max_reached — scale_up requested but max_containers limit hit
        hold_min_reached — scale_down requested but min_containers limit hit

    Note: Person D's main.py passes all return values through normalise_signal()
    which collapses hold_cooldown, hold_max_reached, hold_min_reached → "hold"
    before calling execute(). Direct callers of engine.evaluate() must handle
    all 6 variants or use normalise_signal().

---

## Rules

1. `request_rate` is always an **integer** (cast with `int()`).
2. `timestamp` is a **string** (not a datetime object) when stored in CSV or JSON.
3. Person B's `predict()` always expects **exactly 10 records** — collector must buffer.
4. Any change to this schema requires agreement from all four team members.
