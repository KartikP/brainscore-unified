"""Opt-in CPU selection for tests in any coordinated repository."""
import os

import pytest


@pytest.fixture(autouse=True)
def requested_cpu_profile(monkeypatch):
    """Keep CPU checks off accelerators; individual device tests may override this."""
    if os.environ.get('UMI_TEST_CPU_ONLY') != '1':
        return
    try:
        import torch
    except ImportError:
        # Core and the OpenPI client checks also run without Torch installed.
        return
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
