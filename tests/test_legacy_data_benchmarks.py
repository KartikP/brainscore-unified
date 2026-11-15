"""The legacy MajajHong2015 and Pereira2018 benchmarks have no unified copy.

They reach every subject through the legacy facade, from the vision and language registries.
"""
import pytest

import brainscore
import brainscore.benchmarks  # noqa: F401  (registers the unified benchmarks)

pytestmark = pytest.mark.unit


def test_no_unified_copy_of_a_legacy_benchmark_is_registered():
    assert not [key for key in brainscore.benchmark_registry
                if key.endswith('-unified') or key.startswith(('MajajHong2015', 'Pereira2018'))]
