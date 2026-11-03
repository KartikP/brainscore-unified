"""Maintained extension code uses only public names from other Brain-Score packages.

A static check: source files are parsed, never imported, so this runs without
Torch or model weights. It covers the code that brings models into the
contract (vision and language adapters, vision's leaderboard-contract helper)
and the unified runtime package. It does not cover legacy model and benchmark
plugins or historical data-packaging scripts, which are kept as they were.
"""
import ast
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PACKAGES = ('brainscore_core', 'brainscore_vision', 'brainscore_language', 'brainscore')

# (package, path inside it); a directory covers every .py file below it.
MAINTAINED = [
    ('brainscore_vision', 'compat'),
    ('brainscore_vision', 'benchmark_helpers/test_db_contract.py'),
    ('brainscore_vision', 'models/clip_vit_b_32_unified/model.py'),
    ('brainscore_vision', 'models/qwen25_vl_3b_unified/model.py'),
    ('brainscore_language', 'compat'),
    ('brainscore', ''),
]


def _private(name):
    return name.startswith('_') and not name.startswith('__')


def private_uses(source, own_package):
    """(line, text) for each private name reached in another Brain-Score package."""
    tree = ast.parse(source)
    found, bound = [], {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [(node.module, [alias.name for alias in node.names])]
        elif isinstance(node, ast.Import):
            modules = [(alias.name, []) for alias in node.names]
        else:
            continue
        for module, names in modules:
            parts = module.split('.')
            if parts[0] not in PACKAGES or parts[0] == own_package:
                continue
            for name in [p for p in parts[1:] if _private(p)] + [n for n in names if _private(n)]:
                found.append((node.lineno, f'{module}: {name}'))
            if isinstance(node, ast.ImportFrom):
                bound.update({alias.asname or alias.name: module for alias in node.names})
            else:
                bound.update({(alias.asname or alias.name.split('.')[0]): alias.name
                              for alias in node.names})
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _private(node.attr):
            base = node.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in bound:
                found.append((node.lineno, f'{bound[base.id]}: {node.attr}'))
    return found


def _maintained_files():
    for package, inside in MAINTAINED:
        spec = find_spec(package)  # a top-level lookup does not import the package
        assert spec is not None and spec.submodule_search_locations, package
        target = Path(list(spec.submodule_search_locations)[0]) / inside
        assert target.exists(), target
        files = [target] if target.is_file() else sorted(target.rglob('*.py'))
        for path in files:
            yield package, path


def test_maintained_extension_code_imports_no_private_names():
    offenders = [f'{path}:{line} {what}'
                 for package, path in _maintained_files()
                 for line, what in private_uses(path.read_text(encoding='utf-8'), package)]
    assert not offenders, 'private cross-package names:\n' + '\n'.join(offenders)


@pytest.mark.parametrize('source', [
    'from brainscore_core.streaming_helpers import _drain_stream_events',
    'from brainscore_core._env_check import describe_env',
    'import brainscore_core._env_check',
    'from brainscore_core.supported_data_standards.brainio import fetch\nfetch._local_data_path',
])
def test_guard_catches_private_names(source):
    assert private_uses(source, 'brainscore_vision')


@pytest.mark.parametrize('source', [
    'from brainscore_core.streaming_helpers import drain_session_inputs',
    'from brainscore_core.environment_check import describe_env',
    'from brainscore_vision.compat import _helper',  # own package
    'self._legacy.digest_text(text)',
])
def test_guard_allows_public_and_own_names(source):
    assert not private_uses(source, 'brainscore_vision')
