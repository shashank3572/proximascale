"""
ProximaScale - Evaluation D4, C1, C2, C6, D2-train and model search: one fair protocol
--------------------------------------------------------------------------------------
train.py stops early on the TEST split (validation_data=(X_test, y_test)),
and evaluate.py's univariate baseline trains for a fixed 20 epochs with no
validation at all. Here every LSTM is trained the same way:

    data        the first 80% of metrics.csv (train.py's split); the test
                split is never touched
    validation  the last 10% of that training period, in time order
    fit         the first 90% of it; scalers are fit on these rows only
    training    the paper's architecture (build_model(): LSTM 64 -> Dropout
                0.2 -> LSTM 32 -> Dropout 0.2 -> Dense 3), Adam 1e-3, MSE,
                batch 32, at most 50 epochs, early stopping on validation
                loss with patience 5, best weights restored
    seeds       each variant is trained once per seed (default 0, 1, 2)

Each variant is a spec (VARIANTS below): its input channels, target (CPU or
the Prophet residual), whether the counterfactual correction is applied,
the input window, which Prophet fit supplies the residual, and an optional
loss weighting. Input channels:

    cpu_percent, memory_percent, request_rate   as in the paper
    yhat        Prophet's value at that row's timestamp
    yhat_f3     Prophet's value 3 steps (90 s) after that row
    resid       CPU minus yhat (the residual history the LSTM is asked to extend)
    d_cpu, d_req   change from the previous reading, within the window

Prophet fits ("committed" is model/saved/proximascale_prophet.pkl, fit on the
raw training split):
    clean       refit on corrected training CPU without post-scaling rows,
                then refit again without rows whose residual is beyond
                3.5 robust sd (spikes), so it models the base load only

Nothing in model/saved/ is overwritten: each run goes to
model/saved/variants/<variant>/seed<N>/ (lstm.h5, feature_scaler.pkl,
target_scaler.pkl, info.json); the clean Prophet to
model/saved/variants/_prophet/clean.pkl.

    python model/train_variants.py --skip-existing             # everything
    python model/train_variants.py --variants hybrid --seeds 0
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from counterfactual import apply_counterfactual_correction  # noqa: E402
from preprocessing import FEATURE_COLS, chronological_split, load_csv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = REPO_ROOT / "data" / "collected" / "metrics.csv"
VARIANTS_DIR = REPO_ROOT / "model" / "saved" / "variants"
PROPHET_DIR = VARIANTS_DIR / "_prophet"
COMMITTED_PROPHET = REPO_ROOT / "model" / "saved" / "proximascale_prophet.pkl"

HORIZON = 3
STEP_S = 30
TRAIN_FRAC = 0.8           # train.py's and prophet_model.py's split
VAL_FRAC = 0.1             # last 10% of the training period
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 5
DEFAULT_SEEDS = (0, 1, 2)
CLEAN_SD = 3.5             # robust-sd cut for the clean Prophet's second fit
SPIKE_WEIGHT = 5.0         # loss weight of windows whose next 90 s exceed SPIKE_LEVEL
SPIKE_LEVEL = 70.0

PAPER = list(FEATURE_COLS)
DEFAULTS = {"target": "residual", "correct": True, "window": 10, "prophet": "committed",
            "weight": None, "units": (64, 32), "loss": "mse", "dropout": 0.2,
            "schedule": None, "conv": False}
HUBER_DELTA = 0.1          # in target-scaler units (about 9 CPU points of residual)


def _spec(inputs, **kw):
    return {**DEFAULTS, "inputs": list(inputs), **kw}


VARIANTS = {
    # the 2x2 of inputs x target, plus the correction ablation
    "hybrid": _spec(PAPER),                                            # ProximaScale (D4)
    "univariate": _spec(["cpu_percent"], target="cpu"),                # C6
    "multi_direct": _spec(PAPER, target="cpu"),                        # C1
    "uni_residual": _spec(["cpu_percent"]),                            # C2
    "hybrid_cf_off": _spec(PAPER, correct=False),                      # D2
    # model search: Prophet + residual LSTM, one change at a time
    "hx_prophet_in": _spec(PAPER + ["yhat", "yhat_f3"]),               # sees Prophet
    "hx_resid_in": _spec(["resid", "memory_percent", "request_rate", "yhat_f3"]),
    "hx_clean": _spec(PAPER, prophet="clean"),
    "hx_window20": _spec(PAPER, window=20),
    "hx_window30": _spec(PAPER, window=30),
    "hx_weighted": _spec(PAPER, weight="spike"),
    "hx_deltas": _spec(PAPER + ["d_cpu", "d_req"]),                    # the user's Refinement 5
    "hx_wide": _spec(PAPER, units=(128, 64)),
    # round 2: combine the validation winners (longer history + Prophet information)
    "hx_w30_prophet": _spec(PAPER + ["yhat", "yhat_f3"], window=30),
    "hx_w30_resid": _spec(["resid", "memory_percent", "request_rate", "yhat_f3"], window=30),
    "hx_w60_prophet": _spec(PAPER + ["yhat", "yhat_f3"], window=60),
    # round 3: training and architecture tweaks on top of the round-2 winner
    "hx3_all": _spec(PAPER + ["yhat", "yhat_f3", "resid"], window=30),
    "hx3_huber": _spec(PAPER + ["yhat", "yhat_f3"], window=30, loss="huber"),
    "hx3_plateau": _spec(PAPER + ["yhat", "yhat_f3"], window=30, schedule="plateau"),
    "hx3_drop10": _spec(PAPER + ["yhat", "yhat_f3"], window=30, dropout=0.1),
    "hx3_conv": _spec(PAPER + ["yhat", "yhat_f3"], window=30, conv=True),
}

DELTAS = {"d_cpu": "cpu_percent", "d_req": "request_rate"}


def _log(message):
    print(f"[ProximaScale] {message}", flush=True)


# ============================================================================
# Inputs (shared with variant_eval.py, so training and scoring build the
# same channels the same way)
# ============================================================================

def window_rows(t_rows, window):
    """Row indices of each window ending at t: shape (n, window)."""
    return np.asarray(t_rows, dtype=int)[:, None] - (window - 1) + np.arange(window)


def build_inputs(names, cpu_win, rows, arrays):
    """Input tensor (n, window, channels). cpu_win is each window's CPU as
    the model sees it (corrected per window at inference, or the corrected
    series at training); arrays holds per-row memory, requests and Prophet
    (yhat, aligned to the same rows as the data)."""
    yhat = arrays["yhat"]
    channels = []
    for name in names:
        if name == "cpu_percent":
            channels.append(cpu_win)
        elif name == "resid":
            channels.append(cpu_win - yhat[rows])
        elif name == "yhat":
            channels.append(yhat[rows])
        elif name == "yhat_f3":
            channels.append(yhat[rows + HORIZON])
        elif name in DELTAS:
            source = cpu_win if DELTAS[name] == "cpu_percent" else arrays[DELTAS[name]][rows]
            channels.append(np.concatenate([np.zeros((len(rows), 1)), np.diff(source, axis=1)],
                                           axis=1))
        else:
            channels.append(arrays[name][rows])
    return np.stack(channels, axis=-1)


def row_channels(names, cpu, arrays, rows):
    """The same channels for single rows (window 1, deltas from the previous
    row): what the feature scaler is fit on, like fit_scaler() on rows."""
    out = []
    for name in names:
        if name == "cpu_percent":
            out.append(cpu[rows])
        elif name == "resid":
            out.append(cpu[rows] - arrays["yhat"][rows])
        elif name == "yhat":
            out.append(arrays["yhat"][rows])
        elif name == "yhat_f3":
            out.append(arrays["yhat"][rows + HORIZON])
        elif name in DELTAS:
            source = cpu if DELTAS[name] == "cpu_percent" else arrays[DELTAS[name]]
            out.append(source[rows] - source[np.maximum(rows - 1, 0)])
        else:
            out.append(arrays[name][rows])
    return np.stack(out, axis=-1)


def split_fit_val(train_df, val_frac=VAL_FRAC):
    """Time-ordered: fit = earliest rows of the training period, val = latest."""
    cut = int(len(train_df) * (1 - val_frac))
    return train_df.iloc[:cut].reset_index(drop=True), train_df.iloc[cut:].reset_index(drop=True)


def make_windows(inputs, target, window=10, horizon=HORIZON):
    """X[i] = inputs[i:i+window], y[i] = target[i+window:i+window+horizon];
    windows never cross the end of the array."""
    n = len(inputs) - window - horizon + 1
    idx = np.arange(n)[:, None]
    X = inputs[idx + np.arange(window)]
    y = target[idx + window + np.arange(horizon)]
    return X.astype(np.float32), y.astype(np.float32)


def period_t_rows(start, stop, window):
    """Last input row of every window whose inputs and targets lie in [start, stop)."""
    return np.arange(start + window - 1, stop - HORIZON)


# ============================================================================
# Prophet
# ============================================================================

def load_prophet(name):
    path = COMMITTED_PROPHET if name == "committed" else PROPHET_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def prophet_yhat(name, timestamps):
    from prophet_model import get_prophet_fitted
    return np.asarray(get_prophet_fitted(load_prophet(name), timestamps), dtype=float)


def fit_clean_prophet(train_df):
    """Prophet on the base load only: corrected CPU, post-scaling rows out,
    then spike rows (robust residual beyond CLEAN_SD) out on a second fit."""
    from prophet_model import get_prophet_fitted, train_prophet
    df = train_df.copy()
    df["cpu_percent"] = apply_counterfactual_correction(df)["cpu_percent_corrected"]
    df = df[~df["post_scaling"].to_numpy(dtype=bool)]
    first = train_prophet(df)
    resid = df["cpu_percent"].to_numpy() - np.asarray(get_prophet_fitted(first, df["timestamp"]))
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
    keep = np.abs(resid - np.median(resid)) <= CLEAN_SD * mad
    model = train_prophet(df[keep])
    PROPHET_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROPHET_DIR / "clean.pkl", "wb") as f:
        pickle.dump(model, f)
    info = {"rows_fit": int(keep.sum()), "rows_dropped_post_scaling": int(
        train_df["post_scaling"].sum()), "rows_dropped_spikes": int((~keep).sum()),
        "robust_sd": float(mad), "cut_sd": CLEAN_SD}
    (PROPHET_DIR / "clean.json").write_text(json.dumps(info, indent=2))
    return info


# ============================================================================
# Training
# ============================================================================

def build(n_features, window, units, dropout=0.2, conv=False, loss="mse"):
    """build_model() for the paper's settings; otherwise the same stack with
    the given units, dropout, an optional causal Conv1D front end and loss."""
    from lstm_model import build_model
    if tuple(units) == (64, 32) and dropout == 0.2 and not conv and loss == "mse":
        return build_model(window_size=window, n_features=n_features)
    from tensorflow import keras
    from tensorflow.keras import layers
    stack = [layers.Input(shape=(window, n_features))]
    if conv:
        stack.append(layers.Conv1D(32, 3, padding="causal", activation="relu"))
    stack += [layers.LSTM(units[0], return_sequences=True), layers.Dropout(dropout),
              layers.LSTM(units[1]), layers.Dropout(dropout), layers.Dense(HORIZON)]
    model = keras.Sequential(stack)
    loss_fn = keras.losses.Huber(delta=HUBER_DELTA) if loss == "huber" else "mse"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn,
                  metrics=["mae"])
    return model


def training_arrays(spec, cpu, arrays, start, stop, feature_scaler, target_scaler):
    w = spec["window"]
    t = period_t_rows(start, stop, w)
    rows = window_rows(t, w)
    X = build_inputs(spec["inputs"], cpu[rows], rows, arrays)
    n, _, k = X.shape
    X = feature_scaler.transform(X.reshape(-1, k)).reshape(n, w, k).astype(np.float32)
    target = cpu - arrays["yhat"] if spec["target"] == "residual" else cpu
    future = t[:, None] + np.arange(1, HORIZON + 1)
    y = target_scaler.transform(target[future].reshape(-1, 1)).reshape(n, HORIZON)
    weight = None
    if spec["weight"] == "spike":
        weight = np.where(cpu[future].max(axis=1) > SPIKE_LEVEL, SPIKE_WEIGHT, 1.0)
    return X, y.astype(np.float32), weight


def train_one(variant, seed, data, out_dir, epochs=EPOCHS):
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

    spec = VARIANTS[variant]
    segments = data[(spec["correct"], spec["prophet"])]
    if not isinstance(segments, list):                  # one dataset
        segments = [segments]
    # several datasets (--data a.csv b.csv): each is its own stretch of rows,
    # windows never cross from one into the next; scalers see every fit row
    feature_rows, target_rows = [], []
    for cpu, arrays, (fit_start, cut, stop) in segments:
        fit_rows = np.arange(fit_start, cut)            # as fit_scaler() on the fit rows
        feature_rows.append(row_channels(spec["inputs"], cpu, arrays, fit_rows))
        target = cpu - arrays["yhat"] if spec["target"] == "residual" else cpu
        target_rows.append(target[fit_rows])
    feature_scaler = MinMaxScaler().fit(np.concatenate(feature_rows))
    target_scaler = MinMaxScaler().fit(np.concatenate(target_rows).reshape(-1, 1))

    def stack(parts):
        X, y, w = zip(*parts)
        return (np.concatenate(X), np.concatenate(y),
                None if w[0] is None else np.concatenate(w))
    X_fit, y_fit, w_fit = stack([training_arrays(spec, cpu, arrays, fs, cut, feature_scaler,
                                                 target_scaler)
                                 for cpu, arrays, (fs, cut, stop) in segments])
    X_val, y_val, w_val = stack([training_arrays(spec, cpu, arrays, cut, stop, feature_scaler,
                                                 target_scaler)
                                 for cpu, arrays, (fs, cut, stop) in segments])
    validation = (X_val, y_val) if w_val is None else (X_val, y_val, w_val)

    tf.keras.utils.set_random_seed(seed)
    model = build(len(spec["inputs"]), spec["window"], spec["units"], spec["dropout"],
                  spec["conv"], spec["loss"])
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE * 2 if spec["schedule"] else PATIENCE,
        restore_best_weights=True)]
    if spec["schedule"] == "plateau":
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5))
    started = time.time()
    history = model.fit(X_fit, y_fit, sample_weight=w_fit, validation_data=validation,
                        epochs=epochs, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0,
                        shuffle=True)
    seconds = time.time() - started
    val_loss = history.history["val_loss"]
    best_epoch = int(np.argmin(val_loss)) + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "lstm.h5")
    for name, scaler in (("feature_scaler", feature_scaler), ("target_scaler", target_scaler)):
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(scaler, f)
    info = {
        "variant": variant, "seed": seed, "spec": {**spec, "units": list(spec["units"])},
        "inputs": spec["inputs"], "target": spec["target"],
        "counterfactual_in_training": spec["correct"],
        "fit_windows": len(X_fit), "val_windows": len(X_val),
        "epochs_run": len(val_loss), "best_epoch": best_epoch,
        "best_val_loss": float(min(val_loss)),
        "train_loss_at_best": float(history.history["loss"][best_epoch - 1]),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "parameters": int(model.count_params()), "train_seconds": round(seconds, 1),
        "protocol": {"train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC, "max_epochs": epochs,
                     "batch_size": BATCH_SIZE, "patience": PATIENCE, "optimizer": "adam 1e-3",
                     "loss": "mse", "restore_best_weights": True},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))
    return info


def prepare_data(full_df, train_len, keys):
    """For each (correct, prophet): the training-period CPU the model sees
    (corrected over the whole training split, as train.py does), per-row
    arrays over the whole file (Prophet is a function of time only), and the
    fit / validation boundaries."""
    train_df = full_df.iloc[:train_len]
    cut = int(train_len * (1 - VAL_FRAC))
    raw = full_df["cpu_percent"].to_numpy(dtype=float).copy()
    corrected = raw.copy()
    corrected[:train_len] = apply_counterfactual_correction(train_df)["cpu_percent_corrected"]
    data, yhats = {}, {}
    for correct, prophet in keys:
        if prophet not in yhats:
            yhats[prophet] = prophet_yhat(prophet, full_df["timestamp"])
        arrays = {"yhat": yhats[prophet],
                  "memory_percent": full_df["memory_percent"].to_numpy(dtype=float),
                  "request_rate": full_df["request_rate"].to_numpy(dtype=float)}
        data[(correct, prophet)] = (corrected if correct else raw, arrays, (0, cut, train_len))
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description="Retrain every LSTM under one protocol.")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out-dir", type=Path, default=VARIANTS_DIR)
    parser.add_argument("--data", type=Path, nargs="+", default=[DATA_CSV],
                        help="training CSV(s) (same timestamps as metrics.csv, so the "
                             "committed Prophet applies), e.g. metrics_varied.csv; "
                             "several are trained on together")
    parser.add_argument("--skip-existing", action="store_true",
                        help="leave runs that already have an info.json")
    args = parser.parse_args(argv)

    import tensorflow as tf
    tf.keras.utils.disable_interactive_logging()

    frames = [load_csv(p) for p in args.data]
    full_df = frames[0]
    _log(f"Training data: {', '.join(str(p) for p in args.data)}")
    train_df, _ = chronological_split(full_df, TRAIN_FRAC)
    if any(VARIANTS[v]["prophet"] == "clean" for v in args.variants) and \
            not (PROPHET_DIR / "clean.pkl").exists():
        _log("Fitting the clean Prophet (base load only)...")
        _log(f"  {fit_clean_prophet(train_df)}")
    keys = sorted({(VARIANTS[v]["correct"], VARIANTS[v]["prophet"]) for v in args.variants})
    datas = [prepare_data(f, len(chronological_split(f, TRAIN_FRAC)[0]), keys) for f in frames]
    data = datas[0] if len(datas) == 1 else {k: [d[k] for d in datas] for k in keys}
    cut = int(len(train_df) * (1 - VAL_FRAC))
    _log(f"Training period {len(train_df)} rows: fit {cut}, validation {len(train_df) - cut} "
         f"(from {train_df['timestamp'].iloc[cut]})")

    for variant in args.variants:
        for seed in args.seeds:
            out = args.out_dir / variant / f"seed{seed}"
            if args.skip_existing and (out / "info.json").exists():
                _log(f"{variant} seed {seed}: exists, skipped")
                continue
            info = train_one(variant, seed, data, out, args.epochs)
            info["data"] = [p.name for p in args.data]
            (out / "info.json").write_text(json.dumps(info, indent=2, default=str))
            _log(f"{variant} seed {seed}: best epoch {info['best_epoch']}/{info['epochs_run']}, "
                 f"val loss {info['best_val_loss']:.6f}, {info['train_seconds']:.0f} s")


if __name__ == "__main__":
    main()
