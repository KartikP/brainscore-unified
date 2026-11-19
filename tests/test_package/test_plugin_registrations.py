"""A plugin states each identifier once, on a literal line, and does no work when it is built.

The checks read every model and benchmark plugin folder and the live registries; they
name no plugin. A new plugin is checked when its folder exists.

- Every identifier in ``model_registry`` and ``benchmark_registry`` is the literal key of a
  top-level registration line in the ``__init__.py`` of a plugin folder. Core's plugin
  runner reads these lines to find the identifiers of a changed plugin; a key made in a
  loop or a function is hidden from it.
- Building a registered benchmark starts no session: a benchmark runs nothing before
  ``score()`` can return N/A (Contract Rules 3). A benchmark that cannot be built here
  (its data or its environment is not on this machine, and the network is blocked) is
  not covered by this check; its own ``test.py`` covers it where the data is.
"""
import ast
import socket
import warnings
from pathlib import Path

import pytest

import brainscore
from brainscore_core.streaming import Session

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


class _SessionStarted(Exception):
    pass


@pytest.mark.parametrize('identifier', sorted(brainscore.benchmark_registry))
def test_building_a_benchmark_starts_no_session(identifier, monkeypatch):
    def no_network(*args, **kwargs):
        raise OSError('the network is blocked in this check')

    def refuse(*args, **kwargs):
        raise _SessionStarted()
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(Session, '__init__', refuse)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            brainscore.benchmark_registry[identifier]()
    except _SessionStarted:
        pytest.fail(f'building {identifier} started a session; run the work in __call__ '
                    'or compute it at first use')
    except Exception as error:  # data or environment not here: not covered
        pytest.skip(f'{identifier} does not build on this machine: {type(error).__name__}')
