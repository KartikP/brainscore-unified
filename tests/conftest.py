"""Shared pytest config for the unified test suite.

Tier taxonomy (markers registered in pyproject.toml):
  * unit         — fast, offline, isolated (no real model weights, no network, no S3)
  * integration  — end-to-end across components (e.g. process() -> benchmark -> score),
                   still offline (synthetic models/fixtures), runnable in CI
  * slow         — needs real model downloads / heavy data / GPU / AWS; run on demand
  * private_access — needs S3 / private resources (often combined with slow)

Default behavior (tests/markers.py): any test WITHOUT an explicit tier marker is treated as `unit`,
so `-m unit` and `-m "not slow"` are meaningful without hand-marking every file.
Files that need a real model or heavy data set `pytestmark = pytest.mark.slow`
(e.g. test_text_wrapper, test_roar_benchmark, test_vlm_vision_wrapper); e2e
glue tests set `@pytest.mark.integration`.

Run tiers:
  pytest -m unit                  # fast, every commit
  pytest -m "unit or integration" # CI tier (offline, no big models)
  pytest -m slow                  # on demand (AWS creds + model downloads)
"""
import os

# Cap the math-library threads before numpy loads: on a shared machine, one thread per core in every
# test process slows small matrix tests by orders of magnitude. A value you set yourself stays.
# Covered: a plain `pytest` run (this file) and the workspace launcher (`brainscore/validation/workspace.py`,
# which sets the same defaults in the environment of its pytest subprocess). Not covered when started on their own:
# `install/run_candidate.sh`, `brainscore/validation/installed.py` and CI; export the variables there.
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                  'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_variable, '4')

# The activation store is off unless a run says otherwise: a plain `pytest` must never write to the
# shared store. The store reads its folder when a wrapper class is imported, so this comes before any
# import of a wrapper. An explicit value wins; a test that needs the store on sets it and points the
# wrapper's storage object at its own folder (see tests/test_model_helpers/test_score_provenance.py).
os.environ.setdefault('RESULTCACHING_DISABLE', '1')

# The tier markers and the CPU fixture come from plugins that pytest registers once by name, also
# when the benchmark plugin folders (brainscore/benchmarks/conftest.py) are collected in the same run.
pytest_plugins = ['brainscore.validation.cpu_profile', 'tests.markers']
