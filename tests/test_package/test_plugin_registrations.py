"""A plugin states each identifier once, on a literal line.

The checks read every model and benchmark plugin folder and the live registries; they
name no plugin. A new plugin is checked when its folder exists.

- Every identifier in ``model_registry`` and ``benchmark_registry`` is the literal key of a
  top-level registration line in the ``__init__.py`` of a plugin folder. Core's plugin
  runner reads these lines to find the identifiers of a changed plugin; a key made in a
  loop or a function is hidden from it.
"""
import ast
from pathlib import Path

import pytest

import brainscore

pytestmark = pytest.mark.unit

PACKAGE = Path(brainscore.__file__).parent


def _stated_keys(kind: str, registry: str) -> set:
    """The literal keys of the top-level ``<registry>[...] = ...`` lines of every plugin of ``kind``."""
    keys = set()
    for init in sorted((PACKAGE / kind).glob('*/__init__.py')):
        if init.parent.name.startswith('_'):
            continue
        for statement in ast.parse(init.read_text()).body:
            if not isinstance(statement, ast.Assign):
                continue
            for target in statement.targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == registry and isinstance(target.slice, ast.Constant)):
                    keys.add(target.slice.value)
    return keys


@pytest.mark.parametrize('kind, registry', [('models', 'model_registry'),
                                            ('benchmarks', 'benchmark_registry')])
def test_every_registered_identifier_is_stated_on_a_literal_line(kind, registry):
    hidden = sorted(set(getattr(brainscore, registry)) - _stated_keys(kind, registry))
    assert not hidden, (
        f'{hidden} are registered but not on a literal line of a plugin __init__.py; write '
        f"one line for each identifier: {registry}['<identifier>'] = <factory>")
