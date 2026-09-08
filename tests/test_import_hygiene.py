"""Criterion-2 guard: ``import brainscore`` must not pull heavy ML deps.

Each plugin __init__ defers ``from .model import get_model`` into its factory,
so importing the package stays light — torch/transformers/cv2/sklearn load
only when a model is actually loaded, not when a vision-only or language-only
user imports the package. Runs in a FRESH subprocess because other tests in
the session may already have imported torch.
"""
import subprocess
import sys

HEAVY = ('torch', 'transformers', 'cv2', 'sklearn')


def test_import_brainscore_does_not_load_heavy_deps():
    code = (
        "import sys, brainscore\n"
        f"heavy = {HEAVY!r}\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, f'import brainscore eagerly loaded {loaded}'\n"
        # registries still populate without importing the plugins' model.py
        "assert len(brainscore.model_registry) == 29, "
        "f'model_registry={len(brainscore.model_registry)} (expected 29)'\n"
        "assert len(brainscore.benchmark_registry) == 34, "
        "f'benchmark_registry={len(brainscore.benchmark_registry)} (expected 34)'\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, '-c', code],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r}\nstderr={r.stderr!r}"
    assert 'OK' in r.stdout


def test_loading_a_light_model_still_works():
    """The deferred factory still resolves: chance-baseline has no heavy deps
    and must load through the normal path."""
    import brainscore
    model = brainscore.load_model('chance-baseline')
    assert model is not None
