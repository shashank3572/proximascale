import numpy as np

def detect_anomaly(values, threshold=2.0):
    """
    Z-score anomaly detection on a sequence of CPU values.

    Args:
        values    : array-like of CPU% readings (typically last 10)
        threshold : Z-score threshold above which a point is anomalous

    Returns:
        bool — True if the latest value is anomalous
    """
    values = np.array(values, dtype=float)
    mean   = np.mean(values)
    std    = np.std(values)

    if std == 0:
        return False

    z_score = abs(values[-1] - mean) / std
    return bool(z_score > threshold)