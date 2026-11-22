"""forecast_metrics.py (evaluation M1-M4, M6, C4, D3): metric definitions.

No TensorFlow needed. The anomaly checks compare the vectorized detectors
against the live functions (anomaly.is_anomaly, detect_anomalies_series and
counterfactual.apply_counterfactual_correction) themselves. The match with
evaluate.py's definitions is enforced when the script runs: it re-derives
Table II and stops unless it equals the dump's own evaluate.py check.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

import anomaly  # noqa: E402
import forecast_metrics as F  # noqa: E402
from counterfactual import apply_counterfactual_correction  # noqa: E402


def test_linear_extrapolation_is_slope_of_last_three_points_clamped():
    cpu = np.array([10.0, 20.0, 30.0, 80.0, 90.0, 98.0, 40.0, 10.0, 5.0])
    pred = F.linear_extrapolation(cpu, [2, 5, 8], horizon=3)
    # 30 + 3 * (30 - 10) / 2 = 60;  98 + 3 * 9 -> 125 -> 100;  5 + 3 * (5 - 40) / 2 -> 0
    np.testing.assert_allclose(pred, [60.0, 100.0, 0.0])


def test_lead_times_follow_evaluate_py():
    t_rows = np.arange(9, 200)
    pred = np.zeros(len(t_rows))
    first_t = t_rows[0]
    for t in (48, 49, 50):          # onset 50: earliest qualifying t = 48 -> 60 s
        pred[t - first_t] = 90.0
    pred[96 - first_t] = 80.0       # onset 100: t = 96 is 4 steps early, does not count...
    pred[99 - first_t] = 80.0       # ...t = 99 does -> 30 s
    pred[146 - first_t] = 99.0      # onset 150: t = 147 would be 90 s; 146 is too early -> 0
    pred[150 - first_t] = 99.0      # alarms at or after the onset never count
    leads = F.lead_times(pred, t_rows, [50, 100, 150], 75.0, horizon=3, interval_s=30)
    np.testing.assert_array_equal(leads, [60.0, 30.0, 0.0])


def test_lead_ceiling_is_ramp_start_to_onset_capped_at_horizon():
    ramp, gap, ceiling = F.lead_ceiling(np.array([12, 41, 95]), [10, 40, 90], 3, 30)
    np.testing.assert_array_equal(ramp, [10, 40, 90])
    np.testing.assert_array_equal(gap, [2, 1, 5])
    np.testing.assert_array_equal(ceiling, [60, 30, 90])


def test_spike_onsets_are_first_rows_above_inside_each_spike():
    cpu = np.full(100, 50.0)
    cpu[10:30] = [60, 70, 82, 79, 81, 83] + [82] * 14     # dips below 80 and comes back
    cpu[60:80] = 76.0                                       # never reaches 80
    onsets, starts = F.spike_onsets_above(cpu, [10, 60], 80.0, spike_len=20)
    np.testing.assert_array_equal(onsets, [12])             # the re-crossing at 14 is ignored
    np.testing.assert_array_equal(starts, [10])


def test_crossing_split_matches_same_direction_within_tolerance():
    truth = np.zeros(60, dtype=bool)
    truth[10:25] = True
    lagged = np.roll(truth, 3)                   # persistence: exactly 3 windows late
    assert F.crossing_split(lagged, truth)["spurious_crossings"] == 0
    assert F.crossing_split(lagged, truth)["missed_crossings"] == 0

    blip = truth.copy()
    blip[40:42] = True                           # a false alarm adds two crossings
    assert F.crossing_split(blip, truth) == {
        "crossings": 4, "true_crossings": 2, "net_excess_crossings": 2,
        "spurious_crossings": 2, "missed_crossings": 0}

    early_end = truth.copy()
    early_end[20:25] = False                     # ends 5 windows early: unmatched both ways
    split = F.crossing_split(early_end, truth)
    assert (split["net_excess_crossings"], split["spurious_crossings"],
            split["missed_crossings"]) == (0, 1, 1)

    none = np.zeros(60, dtype=bool)
    assert F.crossing_split(none, truth)["missed_crossings"] == 2


def test_count_matches_is_one_to_one():
    assert F.count_matches(np.array([10]), np.array([9, 10, 11]), 3) == 1
    assert F.count_matches(np.array([10, 12]), np.array([11]), 3) == 1
    assert F.count_matches(np.array([10, 12]), np.array([8, 15]), 3) == 2


def test_event_metrics_window_counts_and_false_alarm_episodes():
    t_rows = np.arange(9, 109)
    actual = np.full(100, 50.0)
    actual[40:55] = 90.0                         # one event: windows 40-54
    pred = np.full(100, 50.0)
    pred[38:52] = 90.0                           # starts 2 windows early, ends 3 early
    pred[70:72] = 90.0                           # a false alarm episode of 2 windows
    pred[80] = 90.0                              # and another of 1 window
    onset = int(t_rows[40]) + 3                  # the row whose CPU first exceeds 75
    metrics, leads = F.event_metrics(pred, actual, t_rows, [onset], 75.0, 3, 30)
    tp, fp, fn = 12, 2 + 2 + 1, 3
    assert metrics["precision"] == pytest.approx(tp / (tp + fp))
    assert metrics["recall"] == pytest.approx(tp / (tp + fn))
    assert metrics["f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert (metrics["alarm_episodes"], metrics["false_alarms"]) == (3, 2)
    assert metrics["false_alarms_per_day"] == pytest.approx(2 / (100 * 30 / 86400))
    assert leads[0] == 90.0 and metrics["caught_90s"] == 1   # window 40 targets the onset


def test_newey_west_variance_known_values():
    d = np.array([1.0, -1.0, 1.0, -1.0])
    assert F.newey_west_variance(d, 0) == pytest.approx(1.0)
    # gamma1 = -3/4; Bartlett weight at lag 1 of 1: 1/2 -> 1 + 2 * 0.5 * (-0.75)
    assert F.newey_west_variance(d, 1) == pytest.approx(0.25)
    smooth = np.repeat(np.random.default_rng(0).normal(size=200), 5)  # positive autocorrelation
    assert F.newey_west_variance(smooth, 10) > 2 * F.newey_west_variance(smooth, 0)


def test_diebold_mariano_sign_and_power():
    rng = np.random.default_rng(1)
    noise = rng.normal(size=5000)
    stat, p = F.diebold_mariano(noise ** 2, (noise + 0.5) ** 2 + 0.25, lag=2)
    assert stat < 0 and p < 1e-6                 # a is clearly better
    stat, p = F.diebold_mariano(rng.normal(size=5000) ** 2, rng.normal(size=5000) ** 2, lag=2)
    assert p > 0.01                              # same loss distribution: no difference


def test_block_bootstrap_means():
    rng = np.random.default_rng(0)
    values = np.vstack([np.arange(10.0), np.full(10, 3.0)])
    whole = F.block_bootstrap_means(values, block_len=10, n_boot=5, rng=rng)
    np.testing.assert_allclose(whole, np.tile([4.5, 3.0], (5, 1)))   # one block = the series
    many = F.block_bootstrap_means(rng.normal(size=(1, 4000)), 20, 500, rng)
    assert many.shape == (500, 1) and abs(many.mean()) < 0.05


def _series_with_scaling_runs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    cpu = 40 + 5 * np.sin(np.arange(n) / 30) + rng.normal(0, 3, n)
    for start in (50, 150, 250):
        cpu[start:start + 4] += [10, 25, 40, 40]          # a spike...
        cpu[start + 4:start + 10] *= 0.8                  # ...then a post-scaling dip
    cpu[300:320] = 0.0                                    # idle: zero-variance baselines
    cpu[320] = 3.0                                        # a jump below the CPU floor
    post = np.zeros(n, dtype=bool)
    for start in (50, 150, 250):
        post[start + 4:start + 10] = True
    return np.round(cpu, 2), post


def test_decision_time_flags_match_predict_load_pipeline():
    cpu, post = _series_with_scaling_runs()
    t_rows = np.arange(9, len(cpu))
    values = F.corrected_window_cpu(cpu, post, t_rows, window=10)
    expected = np.array([apply_counterfactual_correction(pd.DataFrame(
        {"cpu_percent": cpu[t - 9:t + 1], "post_scaling": post[t - 9:t + 1]}))
        ["cpu_percent_corrected"].to_numpy() for t in t_rows])
    np.testing.assert_allclose(values, expected, atol=1e-9)
    for threshold in (2.0, 2.5, 3.5):
        flags = F.anomaly_flags(values, threshold, min_cpu=5.0)
        live = [anomaly.is_anomaly(window, threshold, min_cpu_percent=5.0) for window in expected]
        np.testing.assert_array_equal(flags, live)
    assert F.anomaly_flags(values, 2.5, 5.0).any()


def test_series_flags_match_detect_anomalies_series():
    cpu, _ = _series_with_scaling_runs(seed=3)
    min_cpu = anomaly._configured_min_cpu()
    rows = np.arange(10, len(cpu))
    values = cpu[F.window_matrix(rows, 11)]
    for threshold in (2.0, 3.0):
        live = anomaly.detect_anomalies_series(cpu, window_size=10, threshold=threshold)
        np.testing.assert_array_equal(F.anomaly_flags(values, threshold, min_cpu), live[10:])


def test_detector_scores_against_spikes():
    regimes = np.full(200, "normal", dtype=object)
    for start in (10, 100):
        regimes[start:start + 3] = "ramp"
        regimes[start + 3:start + 17] = "hold"
        regimes[start + 17:start + 20] = "decay"
        regimes[start + 20:start + 30] = "post_scaling"
    scores = F.detector_scores([5, 11, 15, 35], 200, regimes, [10, 100], 30, 20)
    assert scores["precision"] == pytest.approx(0.5)
    assert (scores["recall"], scores["spikes_flagged"], scores["delay_median_s"]) == (0.5, 1, 30)
    assert (scores["false_flags_normal"], scores["false_flags_post_scaling"]) == (1, 1)
    assert (scores["hits_ramp"], scores["hits_hold"]) == (1, 1)
    assert scores["false_flags_per_day"] == pytest.approx(2 / (200 * 30 / 86400))
