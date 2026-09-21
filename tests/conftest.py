"""Shared pytest config for the unified test suite.

Tier taxonomy (markers registered in pyproject.toml):
  * unit         — fast, offline, isolated (no real model weights, no network, no S3)
  * integration  — end-to-end across components (e.g. process() -> benchmark -> score),
                   still offline (synthetic models/fixtures), runnable in CI
  * slow         — needs real model downloads / heavy data / GPU / AWS; run on demand
  * private_access — needs S3 / private resources (often combined with slow)

Default behavior: any test WITHOUT an explicit tier marker is treated as `unit`,
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

import pytest

_TIERS = ('unit', 'integration', 'slow')


def pytest_collection_modifyitems(config, items):
    for item in items:
        if not any(item.get_closest_marker(t) for t in _TIERS):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def requested_cpu_profile(monkeypatch):
    """Keep CPU qualification off accelerators advertised by hosted runners.

    This opt-in applies only to tests. Device-selection tests can still replace
    availability within their own fixture; runtime device selection is unchanged.
    """
    if os.environ.get('UMI_TEST_CPU_ONLY') != '1':
        return
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
