"""
dashboard/shap_explain.py
Person D — Real-time SHAP explainability.

Given Person B's trained Keras LSTM and the current 10-step input
window, compute per-feature attribution (% contribution) for the
prediction that triggered a scaling decision.

NOTE: the LSTM predicts the scaled RESIDUAL of the Prophet forecast
(residual stacking), so this explains what pushed the residual forecast
(the farthest horizon step), not Prophet's seasonal component.

Uses shap.KernelExplainer (model-agnostic). shap.DeepExplainer does not
work with Keras 3 models (TF 2.21 / shap 0.51): it raises
"LookupError: gradient" inside grad_graph.

Features (fixed order, must match Person B's training):
    [cpu_percent, memory_percent, request_rate]

Usage:
    explainer = ShapExplainer(model, X_train_sample)
    pct = explainer.explain(X_input)      # X_input shape (1, 10, 3)
    # → {"CPU%": 45.2, "Memory%": 32.1, "Request Rate": 22.7}
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["CPU%", "Memory%", "Request Rate"]


class ShapExplainer:
    def __init__(self, model, X_train_sample, background_size: int = 100,
                 n_background_clusters: int = 10, nsamples: int = 300):
        """
        model          : trained Keras LSTM (Person B); output = residual per horizon step
        X_train_sample : np.ndarray, shape (N, 10, 3), scaled like the training data
        nsamples       : model evaluations per explanation (speed/accuracy trade-off)
        """
        import shap  # imported lazily so the rest of the dashboard
                     # still imports cleanly if shap isn't installed yet

        self.model = model
        self.feature_names = FEATURE_NAMES
        self.nsamples = nsamples

        bg = np.asarray(X_train_sample[:background_size], dtype=np.float32)
        self._window_shape = bg.shape[1:]           # (10, 3)

        def predict_last_step(flat):
            x = np.asarray(flat, dtype=np.float32).reshape(-1, *self._window_shape)
            return np.asarray(self.model.predict(x, verbose=0))[:, -1]

        background = shap.kmeans(bg.reshape(len(bg), -1),
                                 min(n_background_clusters, len(bg)))
        self.explainer = shap.KernelExplainer(predict_last_step, background)
        logger.debug(f"ShapExplainer initialised with {len(bg)} background samples.")

    def explain(self, X_input: np.ndarray) -> dict:
        """
        X_input : np.ndarray, shape (1, 10, 3)
        Returns : dict {feature_name: percentage}
        """
        X_input = np.asarray(X_input, dtype=np.float32)
        if X_input.ndim == 2:
            X_input = X_input[np.newaxis, ...]   # (10,3) → (1,10,3)

        flat = X_input.reshape(1, -1)
        shap_values = self.explainer.shap_values(flat, nsamples=self.nsamples,
                                                 silent=True)

        # Single-output explainer -> array (1, 30); older shap may return a list
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Reshape to (10, 3); average |SHAP| across timesteps per feature -> (3,)
        per_step = np.asarray(shap_values).reshape(self._window_shape)
        feature_importance = np.abs(per_step).mean(axis=0)

        total = feature_importance.sum()
        if total == 0:
            return {name: 0.0 for name in self.feature_names}

        percentages = (feature_importance / total) * 100.0
        return {
            name: round(float(p), 2)
            for name, p in zip(self.feature_names, percentages)
        }


def format_explanation(pct: dict) -> str:
    """Human-readable string for the dashboard / logs."""
    parts = [f"{k}: {v:.1f}%" for k, v in pct.items()]
    return "Why did we scale? " + ", ".join(parts)


# ── Fallback: when no trained model is available yet ─────────────────────────
def mock_explanation() -> dict:
    """
    Used by the dashboard before Person B's model is wired in.
    Returns plausible attribution so the UI can be built and demoed.
    """
    return {"CPU%": 45.0, "Memory%": 32.0, "Request Rate": 23.0}