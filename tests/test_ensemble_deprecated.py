"""model/ensemble.py is kept as a historical record (a fixed-weight ensemble was
tried, then replaced by residual stacking). It must still work, and must NOT be
used by the live pipeline."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

import ensemble  # noqa: E402


def test_ensemble_predict_still_computes_weighted_average():
    out = ensemble.ensemble_predict([10.0, 20.0, 30.0], [0.0, 0.0, 0.0])
    assert out == pytest.approx([10.0 * ensemble.LSTM_WEIGHT,
                                 20.0 * ensemble.LSTM_WEIGHT,
                                 30.0 * ensemble.LSTM_WEIGHT])


def test_ensemble_predict_rejects_length_mismatch():
    with pytest.raises(ValueError):
        ensemble.ensemble_predict([1.0, 2.0], [1.0])


def test_module_is_marked_deprecated():
    assert "DEPRECATED" in ensemble.__doc__


def test_live_pipeline_does_not_import_ensemble():
    pytest.importorskip("tensorflow", reason="TF stack not installed")
    code = (
        "import sys; sys.path.insert(0, 'model');"
        "import model.predict, model.train;"
        "bad = [m for m in sys.modules if m.split('.')[-1] == 'ensemble'];"
        "assert not bad, bad"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
