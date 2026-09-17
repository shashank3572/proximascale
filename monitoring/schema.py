"""
schema.py — Shared data contract for ProximaScale metrics.
Person B's preprocessing.py imports MetricRecord from here.
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MetricRecord:
    timestamp: str
    cpu_percent: float
    memory_percent: float
    request_rate: int
    post_scaling: bool

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "request_rate": self.request_rate,
            "post_scaling": self.post_scaling,
        }

    @classmethod
    def from_csv_row(cls, row: list) -> "MetricRecord":
        return cls(
            timestamp=row[0].strip(),
            cpu_percent=float(row[1]),
            memory_percent=float(row[2]),
            request_rate=int(float(row[3])),
            post_scaling=row[4].strip().lower() == "true",
        )

    @classmethod
    def from_dict(cls, d: dict) -> "MetricRecord":
        return cls(
            timestamp=d["timestamp"],
            cpu_percent=float(d["cpu_percent"]),
            memory_percent=float(d["memory_percent"]),
            request_rate=int(d["request_rate"]),
            post_scaling=bool(d["post_scaling"]),
        )