"""Benchmarks are experiments; data plugins are their source files.

A benchmark reaches its data through ``load_dataset`` / ``load_stimulus_set`` or,
for identifiers and constants, a data plugin's package. It never imports a
plugin's modules (loaders, builders, packaging scripts), so several benchmarks can
share one data plugin and a plugin can change how it stores its files.
"""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

import brainscore
from tests.architecture_support import plugin_test

# The installed package, so this also holds when tests run outside the checkout.
ROOT = Path(brainscore.__file__).resolve().parent
PLUGINS = {p.name for p in (ROOT / 'data').iterdir() if (p / '__init__.py').exists()}


def _data_imports(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if node.level:                       # relative: resolve against the file's package
                package = path.relative_to(ROOT.parent).parent.parts
                module = '.'.join(package[:len(package) - node.level + 1] + ((module,) if module else ()))
            yield node.lineno, module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name


def test_benchmarks_do_not_import_data_plugin_modules():
    offenders = []
    for path in sorted(path for path in (ROOT / 'benchmarks').rglob('*.py') if not plugin_test(path)):
        for line, module in _data_imports(path):
            parts = module.split('.')
            if parts[:2] == ['brainscore', 'data'] and len(parts) > 3 and parts[2] in PLUGINS:
                offenders.append(f'{path.relative_to(ROOT.parent)}:{line} {module}')
    assert not offenders, 'benchmarks import data plugin internals:\n' + '\n'.join(offenders)


def test_a_benchmark_plugin_imports_no_other_benchmark_plugin():
    """Shared benchmark code lives in a shared place (``brainscore.benchmarks``, ``brainscore.metrics``),
    so one plugin can change without moving another's scores."""
    folders = {p.name for p in (ROOT / 'benchmarks').iterdir() if (p / '__init__.py').exists()}
    # Known before this check: the induced-dyslexia benchmark runs the Yeatman2021 benchmark inside its
    # lesion protocol, and the two share test subjects. Open item: build it from the registry instead.
    known = {('induced_dyslexia', 'roar_yeatman2021'), ('roar_yeatman2021', 'induced_dyslexia')}
    offenders = []
    for folder in sorted(folders):
        for path in sorted((ROOT / 'benchmarks' / folder).rglob('*.py')):
            for line, module in _data_imports(path):
                parts = module.split('.')
                if (parts[:2] == ['brainscore', 'benchmarks'] and len(parts) > 2 and parts[2] in folders - {folder}
                        and (folder, parts[2]) not in known):
                    offenders.append(f'{path.relative_to(ROOT.parent)}:{line} {module}')
    assert not offenders, ('a benchmark plugin imports another benchmark plugin; move the shared code to '
                           'brainscore.benchmarks or brainscore.metrics:\n' + '\n'.join(offenders))


def test_every_data_plugin_registers_something():
    import brainscore
    from brainscore import data_registry, stimulus_set_registry
    registered = set(data_registry) | set(stimulus_set_registry)
    assert registered, 'no data plugins registered'
    assert {'LeBel2023-UTS03', 'LeBel2023-UTS03-words', 'EWoK-core-1.0'} <= registered
