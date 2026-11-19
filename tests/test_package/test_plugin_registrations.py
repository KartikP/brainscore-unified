"""A plugin states each identifier once, on a literal line, and does no work when it is built.

The checks read every model and benchmark plugin folder and the live registries; they
name no plugin. A new plugin is checked when its folder exists.

- Every identifier in ``model_registry`` and ``benchmark_registry`` is the literal key of a
  top-level registration line in the ``__init__.py`` of a plugin folder. Core's plugin
  runner reads these lines to find the identifiers of a changed plugin; a key made in a
  loop or a function is hidden from it.
- Building a registered benchmark starts no session: a benchmark runs nothing before
  ``score()`` can return N/A (Contract Rules 5.6). The network is blocked and installs are
  forbidden (``BS_INSTALL_DEPENDENCIES=no``). A build may stop only because its data or its
  environment is absent (``LoadError``, ``FileNotFoundError`` or the blocked network); any
  other error fails the check with its cause. Not seen here: work a build does after its
  absent data would load, work without a ``Session`` and work in a child process; the
  plugin's own ``test.py`` covers that where the data is.
"""
import ast
import contextlib
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


class _NetworkBlocked(OSError):
    pass


def _absent(error) -> bool:
    """Whether ``error``, or an error it came from, says that data or an environment is absent."""
    while error is not None:
        if isinstance(error, (brainscore.LoadError, FileNotFoundError, _NetworkBlocked)):
            return True
        error = error.__cause__ or error.__context__
    return False


def _check_build_starts_no_session(identifier, factory, monkeypatch):
    started = []

    def no_network(*args, **kwargs):
        raise _NetworkBlocked('the network is blocked in this check')

    def refuse(*args, **kwargs):
        started.append(True)
        raise _SessionStarted()
    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'no')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(Session, '__init__', refuse)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            factory()
    except Exception as error:  # noqa: BLE001  judged below, with its cause
        if not started and not _absent(error):
            pytest.fail(f'building {identifier} failed: {type(error).__name__}: {error}. A build '
                        'may stop only because its data or environment is absent')
    assert not started, (f'building {identifier} started a session; run the work in __call__ '
                         'or compute it at first use')


@pytest.mark.parametrize('identifier', sorted(brainscore.benchmark_registry))
def test_building_a_benchmark_starts_no_session(identifier, monkeypatch):
    _check_build_starts_no_session(identifier, brainscore.benchmark_registry[identifier], monkeypatch)


def _session_after(asset):
    def build():
        asset.read_text()
        Session(None, None, None)
    return build


def _session_caught(asset):
    def build():
        try:
            _session_after(asset)()
        except Exception:  # noqa: BLE001  a constructor that hides the refusal
            pass
    return build


def _broken():
    raise NameError("name 'Benchmark' is not defined")


def _absent_data():
    raise FileNotFoundError('the data is not prepared')


@pytest.mark.parametrize('make, fails', [
    (lambda asset: _absent_data, False),
    (lambda asset: _broken, True),
    (_session_after, True),
    (_session_caught, True),
])
def test_the_build_check_passes_only_a_build_that_stops_for_absent_data(make, fails, tmp_path, monkeypatch):
    asset = tmp_path / 'data.txt'
    asset.write_text('supplied')
    outcome = pytest.raises(BaseException) if fails else contextlib.nullcontext()
    with outcome:
        _check_build_starts_no_session('Synthetic', make(asset), monkeypatch)


def test_the_build_check_forbids_installs(monkeypatch):
    seen = []
    from brainscore_core.plugin_management.import_plugin import installation_preference
    monkeypatch.delenv('BS_INSTALL_DEPENDENCIES', raising=False)
    _check_build_starts_no_session('Synthetic', lambda: seen.append(installation_preference()), monkeypatch)
    assert seen == ['no']
