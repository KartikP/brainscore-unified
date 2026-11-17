"""The installer checks the transformers cache operation the code uses."""
import re
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[1] / 'install' / 'setup.sh'


def _cache_check() -> str:
    line = next(line for line in SETUP.read_text().splitlines()
                if 'Transformers cache API' in line)
    return re.search(r'python -c "(.*)"', line).group(1)


def test_cache_check_uses_slice_kv_cache_not_a_removed_method():
    check = _cache_check()
    assert 'slice_kv_cache' in check
    assert 'to_legacy_cache' not in check      # removed in transformers 5, which is supported


def test_cache_check_passes_in_this_environment():
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    exec(_cache_check(), {})
