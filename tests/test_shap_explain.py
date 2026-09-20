"""dashboard.shap_explain.ShapExplainer must work on Keras 3 models
(shap.DeepExplainer does not)."""
import numpy as np
import pytest

pytest.importorskip("shap", reason="shap not installed")

from dashboard.shap_explain import ShapExplainer  # noqa: E402


class _CpuOnlyModel:
    """Fake 'LSTM': output depends only on feature 0 (cpu). 3 horizon columns."""

    def predict(self, x, verbose=0):
        cpu = x[:, :, 0].sum(axis=1, keepdims=True)
        return np.repeat(cpu, 3, axis=1)


def _background(n=20):
    rng = np.random.default_rng(0)
    return rng.random((n, 10, 3)).astype("float32")


def test_attribution_goes_to_the_feature_the_model_uses():
    exp = ShapExplainer(_CpuOnlyModel(), _background(), nsamples=200)
    pct = exp.explain(_background(1) + 0.5)
    assert set(pct) == {"CPU%", "Memory%", "Request Rate"}
    assert pct["CPU%"] > 95.0


def test_percentages_sum_to_100():
    exp = ShapExplainer(_CpuOnlyModel(), _background(), nsamples=200)
    pct = exp.explain(_background(1))
    assert sum(pct.values()) == pytest.approx(100.0, abs=0.1)


def test_works_on_the_committed_keras3_model():
    pytest.importorskip("tensorflow", reason="TF stack not installed")
    from tensorflow.keras.models import load_model
    from model.lstm_model import MODEL_PATH

    model = load_model(MODEL_PATH, compile=False)
    exp = ShapExplainer(model, _background(), nsamples=100)
    pct = exp.explain(_background(1))

    assert set(pct) == {"CPU%", "Memory%", "Request Rate"}
    assert sum(pct.values()) == pytest.approx(100.0, abs=0.1)
