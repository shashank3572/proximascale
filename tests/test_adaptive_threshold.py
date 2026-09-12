import pytest
from decision.adaptive_threshold import AdaptiveThreshold


def test_falls_back_to_static_when_cold():
    t = AdaptiveThreshold(static_upper=75.0, static_lower=30.0)
    for _ in range(5):
        t.update(50)
    assert t.get_thresholds() == (75.0, 30.0)


def test_bounds_track_sustained_load():
    t = AdaptiveThreshold(min_samples=10, k=1.5, min_band=10.0)
    for _ in range(30):
        t.update(80)
    upper, lower = t.get_thresholds()
    # mu=80, sigma=0 → min_band opens (85, 75)
    # upper stays 85 (within [60, 90])
    # lower gets clamped by ceil_lower=45 → 45
    assert upper == 85.0
    assert lower == 45.0


def test_bounds_widen_with_variance():
    t = AdaptiveThreshold(min_samples=10, k=1.5, floor_upper=0, ceil_upper=200,
                          floor_lower=0, ceil_lower=200)
    for v in [40, 60] * 10:
        t.update(v)
    upper, lower = t.get_thresholds()
    assert upper > 60 and lower < 40     # widened past the raw data