"""ProximaScale — central import-order guard (Python startup hook).

THE PROBLEM
    On Windows, TensorFlow 2.16.1 ships libiomp5md.dll (Intel OpenMP) and
    scikit-learn 1.4.2 ships a vendored vcomp140.dll. If scikit-learn
    initialises its OpenMP runtime BEFORE TensorFlow is imported, the process
    dies with an access violation (exit code 139 / 0xC0000005) inside
    `import tensorflow` — before any ProximaScale code runs, so no amount of
    try/except in our own files can catch it. Verified: TF_ENABLE_ONEDNN_OPTS=0,
    KMP_DUPLICATE_LIB_OK=TRUE, OMP_NUM_THREADS=1, TF_USE_LEGACY_KERAS=0 all
    still crash. Only import order fixes it: TensorFlow first, then
    scikit-learn.

WHY THIS FILE (and not one import line per module)
    `site` imports `sitecustomize` automatically at interpreter startup, so a
    copy installed in the venv's site-packages protects EVERY process run with
    that interpreter: `pytest tests`, `python main.py`, `streamlit run
    dashboard/live_plot.py`, `python -m monitoring.collector`, a REPL, or any
    future entry point — no per-file ordering to remember.
    The hook installs a meta-path finder that fires only when something
    actually imports scikit-learn (directly, or through shap/joblib), imports
    TensorFlow just before it, then lets the normal import machinery proceed.
    Processes that never touch scikit-learn pay nothing.

INSTALL (once per interpreter — redo after recreating .venv)
    cp tools/sitecustomize.py .venv/Lib/site-packages/sitecustomize.py

VERIFY
    .venv/Scripts/python.exe -c "import sklearn, tensorflow"   # must exit 0
    .venv/Scripts/python.exe -m pytest tests                   # full suite

NOTE
    Do not put secrets or project paths in here; it runs before everything
    else. Keep it dependency-free apart from TensorFlow itself.
"""
import sys

# Top-level modules that must not be the first to initialise an OpenMP runtime.
_PRELOAD_TENSORFLOW_BEFORE = ("sklearn",)


class _TensorFlowFirstFinder:
    """Imports TensorFlow the first time scikit-learn is imported."""

    def __init__(self, names=_PRELOAD_TENSORFLOW_BEFORE):
        self._names = set(names)
        self._busy = False

    def find_spec(self, fullname, path=None, target=None):
        # Only the top-level package matters: `import sklearn.preprocessing`
        # imports `sklearn` itself first, which routes through here.
        if fullname not in self._names or self._busy:
            return None
        self._busy = True          # TensorFlow's own imports re-enter this
        try:
            import tensorflow  # noqa: F401 — imported for its side effect only
        except ImportError:
            pass                   # no TensorFlow installed: nothing to protect
        finally:
            self._busy = False
        return None                # let the real finders resolve scikit-learn


def install():
    """Idempotently install the guard on sys.meta_path."""
    if not any(isinstance(finder, _TensorFlowFirstFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _TensorFlowFirstFinder())


install()
