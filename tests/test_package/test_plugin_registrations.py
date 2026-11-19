"""A plugin states each identifier once, on a literal line, and does no work when it is built.

The checks read every model and benchmark plugin folder and the live registries; they
name no plugin. A new plugin is checked when its folder exists.

- Each identifier is stated on a literal line of the folder that registers it. The check
  runs the ``__init__.py`` of each plugin folder again against empty registries and records
  what that folder adds; the identifiers it adds must be the identifiers that core's plugin
  runner (``get_plugin_ids``, the one reader) reads in that same file: one
  ``<registry>['<identifier>'] = <factory>`` line for each identifier. A key made in a loop
  or in a function, a key that another folder or a shared module registers, and a line that
  registers nothing all fail. An identifier of the live registry that no folder adds fails
  when its factory is a part of this package. Not seen here: a key that a shared module of
  this package registers with a factory of another package.
- Building a registered benchmark starts no session, opens no network connection and starts
  no child process (an installer or any other program): a benchmark runs nothing before
  ``score()`` can return N/A (Contract Rules 5.6). Each such attempt fails the check, also
  when the build catches the refusal. A build may stop only for an absent asset that a
  Brain-Score loader names: a local asset (``brainscore.data.local.path`` raises
  ``LocalDataMissing``), a data file that is not in the local store (the store's download is
  refused here) or a package or environment that would be installed (the install switch,
  ``installation_preference``, refuses here). Only that error object, raised to the caller,
  is accepted; any other error fails the check with its cause. Not seen here: work a build
  does after its absent data would load (the plugin's own ``test.py`` covers that where the
  data is), and a child process started without ``subprocess`` or ``os.system``.
"""
import contextlib
import json
import os
import socket
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import brainscore
from brainscore.data import local
from brainscore_core.plugin_management import import_plugin
from brainscore_core.plugin_management.parse_plugin_changes import get_plugin_ids
from brainscore_core.streaming import Session
from brainscore_core.supported_data_standards.brainio.fetch import BotoFetcher

pytestmark = pytest.mark.unit

PACKAGE = Path(brainscore.__file__).parent
REGISTRIES = ('model_registry', 'benchmark_registry', 'metric_registry', 'data_registry',
              'stimulus_set_registry', 'data_builder_registry')


def _added_by_each_folder(kind: str, root: Path, package: str, monkeypatch) -> dict:
    """``{folder: identifiers}``: what the ``__init__.py`` of each plugin folder of
    ``root/kind`` adds to the registry of ``kind`` while it runs (against empty registries)."""
    registry = kind.rstrip('s') + '_registry'
    added = {}
    for init in sorted((root / kind).glob('*/__init__.py')):
        folder = init.parent.name
        if folder.startswith('_'):
            continue
        name = f'{package}.{kind}.{folder}'
        with monkeypatch.context() as patch:
            for each in REGISTRIES:
                patch.setattr(brainscore, each, {})
            exec(compile(init.read_text(), str(init), 'exec'),
                 {'__name__': name, '__package__': name, '__file__': str(init),
                  '__path__': [str(init.parent)]})
            added[folder] = list(getattr(brainscore, registry))
    return added


def _unstated(registry, kind: str, root: Path, package: str, monkeypatch) -> list:
    """The ``(folder, identifier)`` pairs where what a folder adds and what the plugin runner
    reads in that folder differ, and ``(None, identifier)`` for a key of ``registry`` with a
    factory of ``package`` that no folder adds."""
    added = _added_by_each_folder(kind, root, package, monkeypatch)
    unstated = []
    for folder, identifiers in added.items():
        try:
            stated = get_plugin_ids(kind, [folder], str(root))
        except Exception:  # noqa: BLE001  the runner cannot read this folder at all
            stated = []
        unstated += [(folder, identifier) for identifier in set(identifiers) ^ set(stated)]
    everywhere = {identifier for identifiers in added.values() for identifier in identifiers}
    for identifier, factory in registry.items():
        module = getattr(getattr(factory, 'func', factory), '__module__', None) or ''
        if identifier not in everywhere and (module == package or module.startswith(f'{package}.')):
            unstated.append((None, identifier))
    return sorted(unstated, key=str)


@pytest.mark.parametrize('kind, registry', [('models', 'model_registry'),
                                            ('benchmarks', 'benchmark_registry')])
def test_every_identifier_is_stated_on_a_line_of_its_own_folder(kind, registry, monkeypatch):
    unstated = _unstated(getattr(brainscore, registry), kind, PACKAGE, 'brainscore', monkeypatch)
    assert not unstated, (
        f'{unstated} (folder, identifier): the folder registers the identifier and core\'s plugin '
        f"runner cannot read it in the folder's __init__.py, or the runner reads it there and "
        f'the folder does not register it; write one line in the folder that registers it, '
        f"for each identifier: {registry}['<identifier>'] = <factory>")


_SHARED = 'def factory():\n    pass\n'
_STAND_INS = {
    'shared factory': {'one': "benchmark_registry['a'] = factory\n",
                       'two': "benchmark_registry['b'] = factory\n"},
    'split line': {'one': "benchmark_registry[\n    'a'] = factory\n"},
    'loop': {'one': "for key in ('a',):\n    benchmark_registry[key] = factory\n"},
    'computed key, outside factory': {
        'one': "benchmark_registry['a'] = factory\n"
               "for key in ('c',):\n    benchmark_registry[key] = partial(json.dumps)\n"},
    'literal borrowed by another folder': {
        'one': "def _unused():\n    benchmark_registry['b'] = factory\n",
        'two': "for key in ('b',):\n    benchmark_registry[key] = factory\n"},
}


@pytest.mark.parametrize('stand_in, unstated', [
    ('shared factory', []),
    ('split line', [('one', 'a')]),
    ('loop', [('one', 'a')]),
    ('computed key, outside factory', [('one', 'c')]),
    ('literal borrowed by another folder', [('one', 'b'), ('two', 'b')]),
])
def test_the_line_check_compares_what_a_folder_adds_with_its_lines(stand_in, unstated, tmp_path,
                                                                    monkeypatch):
    kind = tmp_path / 'standin' / 'benchmarks'
    kind.mkdir(parents=True)
    for package in (kind.parent, kind):
        (package / '__init__.py').write_text('')
    (kind / '_shared.py').write_text(_SHARED)
    header = ('import json\nfrom functools import partial\nfrom brainscore import benchmark_registry\n'
              'from standin.benchmarks._shared import factory\n')
    for folder, lines in _STAND_INS[stand_in].items():
        (kind / folder).mkdir()
        (kind / folder / '__init__.py').write_text(header + lines)
    monkeypatch.syspath_prepend(str(tmp_path))
    live = {}                                           # the registry as the import leaves it
    with monkeypatch.context() as patch:
        patch.setattr(brainscore, 'benchmark_registry', live)
        for folder in _STAND_INS[stand_in]:
            __import__(f'standin.benchmarks.{folder}')
    live['external'] = json.loads                       # a plugin of another package
    try:
        found = _unstated(live, 'benchmarks', kind.parent, 'standin', monkeypatch)
        assert set(unstated) <= set(found) and bool(found) == bool(unstated), found
    finally:
        for name in [name for name in sys.modules if name.startswith('standin')]:
            del sys.modules[name]


class _Refused(Exception):
    """What the build check raises in place of a session, a download, a child process or an
    install."""


def _check_build_starts_no_session(identifier, factory, monkeypatch):
    attempts, absent = [], []

    def refuse(what):
        def stand_in(*args, **kwargs):
            attempts.append(what)
            raise _Refused(f'{what} is refused in this check')
        return stand_in

    def absent_here(what):
        def stand_in(*args, **kwargs):
            absent.append(_Refused(f'{what} is absent here'))
            raise absent[-1]
        return stand_in

    def local_path(name, _path=local.path):
        try:
            return _path(name)
        except local.LocalDataMissing as error:
            absent.append(error)
            raise
    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'no')
    monkeypatch.setattr(Session, '__init__', refuse('a session'))
    monkeypatch.setattr(socket.socket, 'connect', refuse('a network connection'))
    monkeypatch.setattr(subprocess.Popen, '__init__', refuse('a child process'))
    monkeypatch.setattr(os, 'system', refuse('a child process'))
    monkeypatch.setattr(local, 'path', local_path)
    monkeypatch.setattr(BotoFetcher, 'download_boto', absent_here('a data file of the store'))
    monkeypatch.setattr(import_plugin, 'installation_preference',
                        absent_here('a package or environment to install'))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            factory()
    except Exception as error:  # noqa: BLE001  judged below, with its cause
        if not attempts and not any(error is each for each in absent):
            pytest.fail(f'building {identifier} failed: {type(error).__name__}: {error}. A build '
                        'may stop only for an absent asset that a Brain-Score loader names '
                        '(brainscore.data.local.path, the data store, the install switch)')
    assert not attempts, (f'building {identifier} tried {sorted(set(attempts))}; run the work in '
                          '__call__ or compute it at first use')


@pytest.mark.parametrize('identifier', sorted(brainscore.benchmark_registry))
def test_building_a_benchmark_starts_no_session(identifier, monkeypatch):
    _check_build_starts_no_session(identifier, brainscore.benchmark_registry[identifier], monkeypatch)


def _needs_absent_data(asset):
    return lambda: local.path(asset)


def _needs_absent_environment(asset):
    def build():
        if import_plugin.installation_preference() == 'no':
            raise brainscore.LoadError('the environment is absent and installs are refused')
    return build


def _session_after(asset):
    def build():
        local.path(asset)
        Session(None, None, None)
    return build


def _session_caught(asset):
    def build():
        try:
            _session_after(asset)()
        except Exception:  # noqa: BLE001  a constructor that hides the refusal
            pass
    return build


def _download_refused(asset):
    def build():
        try:
            socket.socket().connect(('127.0.0.1', 9))
        except OSError as error:
            raise FileNotFoundError('the data is not here') from error
    return build


def _download_caught(asset):
    def build():
        with contextlib.suppress(Exception):
            socket.socket().connect(('127.0.0.1', 9))
    return build


def _child_process_caught(asset):
    def build():
        with contextlib.suppress(Exception):
            subprocess.run([sys.executable, '-c', 'pass'], check=True)
    return build


def _broken_after_a_missing_file(asset):
    def build():
        try:
            open(Path(local.data_home()) / 'not-there')
        except OSError:
            raise NameError("name 'Benchmark' is not defined")
    return build


def _packaging_fault(asset):
    def build():
        raise brainscore.LoadError('this installation lacks the pinned packages (a packaging fault)')
    return build


@pytest.mark.parametrize('make, fails', [
    (_needs_absent_data, False),
    (_needs_absent_environment, False),
    (_session_after, True),
    (_session_caught, True),
    (_download_refused, True),
    (_download_caught, True),
    (_child_process_caught, True),
    (_broken_after_a_missing_file, True),
    (_packaging_fault, True),
])
def test_the_build_check_passes_only_a_build_that_stops_for_a_named_absent_asset(
        make, fails, tmp_path, monkeypatch):
    monkeypatch.setenv(local.HOME_VARIABLE, str(tmp_path))
    monkeypatch.setitem(local.REGISTRY, 'stand-in', local.LocalAsset(
        name='stand-in', kind='stimuli', why_local='a stand-in', source='nowhere', obtain='nothing'))
    if make is _session_after or make is _session_caught:
        (tmp_path / 'stand-in').write_text('supplied')
    outcome = pytest.raises(BaseException) if fails else contextlib.nullcontext()
    with outcome:
        _check_build_starts_no_session('Synthetic', make('stand-in'), monkeypatch)
