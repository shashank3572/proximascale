# ProximaScale — Data Schema Contract

> **This document is the team contract. Nobody changes the schema without a team discussion.**
>
> Written by Person D (Week 2). Referenced by Person A (produces metrics), Person B (consumes metrics / produces forecasts), and Person C (consumes forecasts / produces scaling decisions).

---

## Metric Record — JSON / Dict Format

```json
{
  "timestamp":      "2024-01-15T14:32:00",
  "cpu_percent":    67.4,
  "memory_percent": 52.1,
  "request_rate":   143,
  "post_scaling":   false
}
```

| Field            | Type    | Unit / Range     | Notes                                                                   |
| ---------------- | ------- | ---------------- | ----------------------------------------------------------------------- |
| `timestamp`      | string  | ISO-8601         | `YYYY-MM-DDTHH:MM:SS`, no timezone                                      |
| `cpu_percent`    | float   | 0.0–100.0        | Container CPU utilisation from Docker SDK / cgroup metrics              |
| `memory_percent` | float   | 0.0–100.0        | Container memory utilisation from Docker SDK                            |
| `request_rate`   | integer | 0–∞              | HTTP request count/rate measured over the 30-second monitoring interval |
| `post_scaling`   | bool    | `true` / `false` | Counterfactual tag marking observations collected after a scaling event |

---

## CSV File Format

**File:** `data/collected/metrics.csv`

```csv
timestamp,cpu_percent,memory_percent,request_rate,post_scaling
2024-01-15T14:32:00,67.4,52.1,143,false
2024-01-15T14:32:30,72.1,54.3,201,false
```

* Header row is always present and is written by `monitoring/storage.py`.
* One row is written per monitoring polling interval.
* **Current Phase 2 polling interval: 30 seconds.**
* The 10 most recent records represent a **5-minute historical window**.
* `request_rate` is measured over the configured 30-second monitoring interval.
* Rows are parsed by `monitoring/schema.py → MetricRecord.from_csv_row()`.

---

## Python Class

```python
from monitoring.schema import MetricRecord

# From dict / JSON payload
record = MetricRecord.from_dict(d)

# From CSV row (list of strings)
record = MetricRecord.from_csv_row(row)

# To dict
d = record.to_dict()
```

`MetricRecord` is the canonical representation of a monitoring observation exchanged between the monitoring and prediction components.

---

## Interface Between Modules

```text
Person A — App + Monitoring
    monitoring/collector.py
        ↓
    produces MetricRecord objects
        ↓
    monitoring/storage.py
        ↓
    data/collected/metrics.csv


Person B — LSTM + ML
    model/predict.py
        ↑
    consumes exactly 10 chronological metric records
        ↑
    main.py → storage.read_last_n(10)

    call:
        predict_load(records)

    returns:
        predicted_cpu: float
        upper_bound: float
        anomaly_flag: bool

    predicted_cpu
        → final residual-stacked CPU forecast
          (Prophet + LSTM residual)
          for the configured forecast horizon
          (approximately 90 seconds ahead)

    upper_bound
        → upper prediction bound for the same
          forecast horizon

    anomaly_flag
        → True when the prediction pipeline
          detects an anomaly / spike


Person C — Decision + Actuator
    decision/engine.py
        ↑
    consumes:
        predicted_cpu: float
        upper_bound: float
        anomaly_flag: bool

    call:
        engine.evaluate(
            predicted_cpu,
            upper_bound=None,
            anomaly_flag=False
        )

    returns:
        "scale_up" | "scale_down" | "hold"

    and updates:
        last_reason
```

---

## Decision Return Values

| Return value | Meaning                                                    |
| ------------ | ---------------------------------------------------------- |
| `scale_up`   | Predicted / assessed CPU load requires adding a container  |
| `scale_down` | Predicted / assessed CPU load permits removing a container |
| `hold`       | No scaling action is required                              |

### `last_reason` Values

Expected reason values include:

```text
in_band
cooldown
max_reached
min_reached
anomaly
upper_bound_risk
cpu_high
cpu_low
*_no_actuator
*_error
```

`main.py` no longer requires the previous `normalise_signal` variants, although defensive signal normalisation may still be performed where necessary.

---

## Post-Scaling Counterfactual Tag

When Person C's actuator performs a scaling event, the event is recorded in:

```text
data/scaling_events.json
```

The monitoring module uses this event information to mark subsequent metric records with:

```text
post_scaling = true
```

for the configured post-scaling period.

This tag identifies CPU behaviour that may have been artificially affected by a recent scaling action. Person B's prediction pipeline can therefore distinguish an actuator-induced CPU change from a genuine change in workload demand.

Conceptually:

```text
Scaling event
      ↓
Container capacity changes
      ↓
CPU may drop artificially
      ↓
post_scaling = true
      ↓
Prediction pipeline accounts for the
recent scaling intervention
```

---

## Rules

1. `request_rate` is always an **integer** and must be cast using `int()` where required.
2. `timestamp` is stored as a **string**, not a `datetime` object, when persisted to CSV or exchanged through JSON/dictionaries.
3. Metric records must remain in **chronological order** when passed to the prediction pipeline.
4. `predict_load()` expects **exactly 10 metric records** for the current Phase 2 prediction pipeline.
5. `main.py` obtains these records using `storage.read_last_n(10)`.
6. The current Phase 2 monitoring interval is **30 seconds**.
7. Therefore, 10 records represent approximately **5 minutes of historical data**.
8. The monitoring polling interval and the request-rate measurement interval are aligned at **30 seconds** unless the implementation explicitly changes the measurement method.
9. `cpu_percent` and `memory_percent` refer to **container-level resources**, not host-machine resources.
10. Any change to the metric fields, field types, meanings, prediction input contract, monitoring interval, or decision-engine interface requires agreement from **all four team members**.
11. Code implementing the schema must remain consistent with this document. If implementation and documentation diverge, the discrepancy must be resolved before changing the contract.
12. The Phase 1 implementation used a shorter monitoring interval, but the current Phase 2 contract uses **30-second sampling** to match the current prediction pipeline.
