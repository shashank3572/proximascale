"""
ProximaScale - Stacked forecaster v2 (Prophet + residual LSTM, model search winner)
-----------------------------------------------------------------------------------
The model chosen by the evaluation's model search (train_variants.py,
variant "hx_w30_prophet", picked on the validation period only):

    inputs   30 readings (15 min) of corrected CPU, memory and requests, plus
             Prophet's value at each reading's timestamp (yhat) and 90 s after
             it (yhat_f3)
    target   the Prophet residual for t+1..t+3, as the paper's design
    output   Prophet + residual, the same as predict_load() has always returned
    members  the 3 training seeds, pooled: each runs n_passes / 3 MC Dropout
             passes and all passes are pooled into one mean and sigma, so a
             decision still costs 30 passes in total

A bundle directory holds config.json and one folder per member with
lstm.h5, feature_scaler.pkl and target_scaler.pkl (train_variants.py's
output, copied unchanged). predict.py serves it live; eval_dump.py --bundle
runs the same code over every test window.
"""

import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from counterfactual import apply_counterfactual_correction  # noqa: E402
from train_variants import HORIZON, STEP_S, build_inputs  # noqa: E402

STD_MULTIPLIER = 2.0          # uncertainty.py's mean + 2 sigma (when uncalibrated)
PROPHET_CHANNELS = {"yhat", "yhat_f3", "resid"}


def read_config(bundle_dir):
    return json.loads((Path(bundle_dir) / "config.json").read_text(encoding="utf-8"))


class StackedForecaster:
    def __init__(self, bundle_dir, prophet_model, seed=None):
        """Loads every member. seed (optional) seeds Python/NumPy/TF BEFORE the
        models are built: Keras 3 fixes each Dropout layer's random state then,
        so this is what makes a run reproducible."""
        import tensorflow as tf
        self.bundle_dir = Path(bundle_dir)
        self.config = read_config(bundle_dir)
        self.inputs = list(self.config["inputs"])
        self.window = int(self.config["window"])
        self.correct = bool(self.config.get("correct", True))
        self.prophet_model = prophet_model
        if seed is not None:
            tf.keras.utils.set_random_seed(seed)
        self.members = []
        for name in self.config["members"]:
            d = self.bundle_dir / name
            model = tf.keras.models.load_model(d / "lstm.h5", compile=False)
            with open(d / "feature_scaler.pkl", "rb") as f:
                feature_scaler = pickle.load(f)
            with open(d / "target_scaler.pkl", "rb") as f:
                target_scaler = pickle.load(f)
            if tuple(model.input_shape[1:]) != (self.window, len(self.inputs)):
                raise ValueError(f"{d} expects {model.input_shape[1:]}, config says "
                                 f"({self.window}, {len(self.inputs)})")
            self.members.append((model, feature_scaler, target_scaler))

    # ------------------------------------------------------------------ inputs
    def window_inputs(self, df):
        """One live window (DataFrame, oldest -> newest, exactly self.window
        rows with timestamp, cpu_percent, memory_percent, request_rate and
        optional post_scaling). Returns (raw inputs (window, k), Prophet's
        forecast for t+1..t+3, the corrected CPU)."""
        if len(df) != self.window:
            raise ValueError(f"window must have {self.window} readings, got {len(df)}")
        df = df.reset_index(drop=True).copy()
        if "post_scaling" not in df.columns:
            df["post_scaling"] = False
        cpu = df["cpu_percent"].to_numpy(dtype=float)
        if self.correct:
            cpu = apply_counterfactual_correction(df)["cpu_percent_corrected"].to_numpy(float)
        stamps = pd.to_datetime(df["timestamp"])
        last = stamps.iloc[-1]
        future = [last + pd.Timedelta(seconds=STEP_S * k) for k in range(1, HORIZON + 1)]
        from prophet_model import get_prophet_fitted
        yhat = np.asarray(get_prophet_fitted(self.prophet_model, list(stamps) + future),
                          dtype=float)                       # rows 0..w-1, then t+1..t+3
        arrays = {"yhat": yhat,
                  "memory_percent": df["memory_percent"].to_numpy(dtype=float),
                  "request_rate": df["request_rate"].to_numpy(dtype=float)}
        rows = np.arange(self.window)[None, :]
        X = build_inputs(self.inputs, cpu[None, :], rows, arrays)[0]
        return X, yhat[self.window:self.window + HORIZON], cpu

    def history_inputs(self, full_df, t_abs):
        """window_inputs() for many windows at once: the windows ending at rows
        t_abs of full_df (metrics.csv order), reading history before them as
        the live loop would. Returns (raw inputs (n, window, k), Prophet's
        forecast (n, HORIZON))."""
        from forecast_metrics import corrected_window_cpu
        from prophet_model import get_prophet_fitted
        from train_variants import window_rows
        t_abs = np.asarray(t_abs, dtype=int)
        w = self.window
        if t_abs.min() - (w - 1) < 0 or t_abs.max() + HORIZON >= len(full_df):
            raise ValueError("windows need w-1 rows of history and 3 rows after")
        lo, hi = t_abs.min() - (w - 1), t_abs.max() + HORIZON + 1
        cpu_full = full_df["cpu_percent"].to_numpy(dtype=float)
        post = full_df["post_scaling"].to_numpy(dtype=bool)
        rows = window_rows(t_abs, w)
        cpu = (corrected_window_cpu(cpu_full, post, t_abs, w) if self.correct
               else cpu_full[rows])
        yhat = np.full(len(full_df), np.nan)
        yhat[lo:hi] = get_prophet_fitted(self.prophet_model, full_df["timestamp"].iloc[lo:hi])
        arrays = {"yhat": yhat,
                  "memory_percent": full_df["memory_percent"].to_numpy(dtype=float),
                  "request_rate": full_df["request_rate"].to_numpy(dtype=float)}
        X = build_inputs(self.inputs, cpu, rows, arrays)
        future = yhat[t_abs[:, None] + np.arange(1, HORIZON + 1)]
        return X, future

    def scale(self, X_raw, member=0):
        """Raw inputs (n, window, k) scaled with one member's feature scaler."""
        scaler = self.members[member][1]
        n, w, k = X_raw.shape
        return scaler.transform(X_raw.reshape(-1, k)).reshape(n, w, k).astype(np.float32)

    # ---------------------------------------------------------------- forecast
    def forecast(self, X_raw, prophet_future, n_passes=30):
        """X_raw (n, window, k), prophet_future (n, HORIZON). Each member runs
        ceil(n_passes / members) MC Dropout passes; all passes are pooled.
        Returns mean, sigma and mean + 2 sigma, each (n, HORIZON) in CPU %,
        before any 0-100 clamp."""
        X_raw = np.asarray(X_raw, dtype=float)
        prophet_future = np.asarray(prophet_future, dtype=float).reshape(len(X_raw), HORIZON)
        per_member = max(1, math.ceil(n_passes / len(self.members)))
        pooled = []
        for i, (model, _, target_scaler) in enumerate(self.members):
            X = self.scale(X_raw, i)
            if len(X) == 1:                               # live: one batched call
                out = model(np.repeat(X, per_member, axis=0), training=True).numpy()
                passes = out[:, None, :]
            else:
                passes = np.stack([model(X, training=True).numpy()
                                   for _ in range(per_member)])
            resid = (passes - target_scaler.min_[0]) / target_scaler.scale_[0]
            pooled.append(resid + prophet_future[None])
        pooled = np.concatenate(pooled)
        mean, sigma = pooled.mean(axis=0), pooled.std(axis=0)
        return {"mean": mean, "sigma": sigma, "upper": self.upper(mean, sigma)}

    def upper(self, mean, sigma):
        """The upper bound: config.json's calibrated one-sided bound (fit on
        validation by tune_decisions.py), else mean + 2 sigma. "binned" adds a
        margin per sigma quintile, "margin" a constant, "scaled" k * sigma."""
        cal = self.config.get("upper_bound")
        if not cal:
            return mean + STD_MULTIPLIER * sigma
        if cal["method"] == "scaled":
            return mean + cal["k"] * sigma
        if cal["method"] == "margin":
            return mean + cal["q"]
        bins = np.digitize(sigma, np.asarray(cal["edges"]))
        return mean + np.asarray(cal["q"])[bins]
