"""Committed model artifacts must load in the pinned environment and match
the residual-stacking contract. Guards against silent fallback in predict_load()."""
import pickle
import sys
from pathlib import Path

import pytest

pytest.importorskip("tensorflow", reason="TF stack not installed")
pytest.importorskip("prophet", reason="Prophet not installed")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from lstm_model import MODEL_PATH, WINDOW_SIZE, N_FEATURES, HORIZON  # noqa: E402

SAVED = Path(MODEL_PATH).parent


def test_model_path_exists():
    assert Path(MODEL_PATH).exists(), f"missing LSTM artifact: {MODEL_PATH}"


def test_lstm_loads_and_matches_contract():
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH, compile=False)
    assert model.input_shape == (None, WINDOW_SIZE, N_FEATURES)
    assert model.output_shape == (None, HORIZON)


def test_residual_scaler_is_single_column():
    with open(SAVED / "scaler_residual.pkl", "rb") as f:
        assert pickle.load(f).n_features_in_ == 1


def test_prophet_artifact_loads():
    from prophet_model import load_prophet_model
    assert load_prophet_model() is not None
