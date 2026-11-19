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
        # the registries populate without importing the plugins' model.py
        "assert brainscore.model_registry and brainscore.benchmark_registry\n"
        # no plugin __init__ imports a plugin's model.py or benchmark.py (any plugin)
        "eager = sorted(m for m in sys.modules if m.startswith(('brainscore.models.', "
        "'brainscore.benchmarks.')) and m.rsplit('.', 1)[1] in ('model', 'benchmark'))\n"
        "assert not eager, f'import brainscore imported {eager}; import model.py or "
        "benchmark.py inside the factory'\n"
        "print('OK')\n"
    )
    _run_fresh(code)


def test_import_brainscore_loads_neither_the_coggym_harness_nor_the_ewok_preparation():
    """Named regressions: these two modules once loaded at import."""
    _run_fresh(
        "import sys, brainscore\n"
        "assert 'brainscore.harnesses.coggym' not in sys.modules\n"
        "assert 'brainscore.data.ewok.data_packaging.prepare' not in sys.modules\n"
        "print('OK')\n")


def _run_fresh(code: str) -> None:
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
