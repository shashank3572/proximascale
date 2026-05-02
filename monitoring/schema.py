"""
schema.py — Shared data contract for ProximaScale metrics.
Person B's preprocessing.py imports MetricRecord from here.
"""
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class MetricRecord:
    timestamp: str        # ISO-8601 string, e.g. "2024-01-15T14:32:00"
    cpu_percent: float    # 0.0 – 100.0
    memory_percent: float # 0.0 – 100.0
    request_rate: int     # requests in the last 60-second window

    def to_dict(self) -> dict:
        """Returns the canonical team-agreed JSON/dict format."""
        return {
            "timestamp":      self.timestamp,
            "cpu_percent":    self.cpu_percent,
            "memory_percent": self.memory_percent,
            "request_rate":   self.request_rate,
        }

    @classmethod
    def from_csv_row(cls, row: list) -> "MetricRecord":
        """
        Parses one CSV row (list of strings) into a MetricRecord.
        Expects order: timestamp, cpu_percent, memory_percent, request_rate
        """
        return cls(
            timestamp=row[0].strip(),
            cpu_percent=float(row[1]),
            memory_percent=float(row[2]),
            request_rate=int(float(row[3])),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "MetricRecord":
        return cls(
            timestamp=d["timestamp"],
            cpu_percent=float(d["cpu_percent"]),
            memory_percent=float(d["memory_percent"]),
            request_rate=int(d["request_rate"]),
        )
