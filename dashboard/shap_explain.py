"""
dashboard/shap_explain.py
Person D — Real-time SHAP explainability.

Given Person B's trained Keras LSTM and the current 10-step input
window, compute per-feature attribution (% contribution) for the
prediction that triggered a scaling decision.

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
    def __init__(self, model, X_train_sample, background_size: int = 100):
        """
        model          : trained Keras LSTM (Person B)
        X_train_sample : np.ndarray, shape (N, 10, 3)
        """
        import shap  # imported lazily so the rest of the dashboard
                     # still imports cleanly if shap isn't installed yet

        self.model = model
        self.feature_names = FEATURE_NAMES

        bg = np.asarray(X_train_sample[:background_size], dtype=np.float32)
        self.explainer = shap.DeepExplainer(model, bg)
        logger.info(f"ShapExplainer initialised with {len(bg)} background samples.")

    def explain(self, X_input: np.ndarray) -> dict:
        """
        X_input : np.ndarray, shape (1, 10, 3)
        Returns : dict {feature_name: percentage}
        """
        X_input = np.asarray(X_input, dtype=np.float32)
        if X_input.ndim == 2:
            X_input = X_input[np.newaxis, ...]   # (10,3) → (1,10,3)

        shap_values = self.explainer.shap_values(X_input)

        # shap_values may be a list (multi-output) or a single array
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # shap_values shape: (1, 10, 3)
        # Average |SHAP| across timesteps per feature → (3,)
        feature_importance = np.abs(shap_values[0]).mean(axis=0)

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