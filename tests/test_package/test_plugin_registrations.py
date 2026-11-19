"""A plugin states each identifier once, on a literal line, and does no work when it is built.

The checks read every model and benchmark plugin folder and the live registries; they
name no plugin. A new plugin is checked when its folder exists.

- Every identifier that a shipped plugin folder registers is read from the ``__init__.py``
  of that folder by core's plugin runner (``get_plugin_ids``, the one reader): one
  ``<registry>['<identifier>'] = <factory>`` line for each identifier. A key made in a loop or
  in a function, or stated in another folder, is hidden from the runner. The folder of an
  identifier is the folder of its factory's source; an identifier whose factory is in a
  shared module of the package is checked against every folder; a plugin of another
  package is not checked here.
- Building a registered benchmark starts no session: a benchmark runs nothing before
  ``score()`` can return N/A (Contract Rules 5.6). The network is blocked and installs are
  forbidden (``BS_INSTALL_DEPENDENCIES=no``). A build may stop only because its data or its
  environment is absent (``LoadError``, ``FileNotFoundError`` or the blocked network); any
  other error fails the check with its cause. Not seen here: work a build does after its
  absent data would load, work without a ``Session`` and work in a child process; the
  plugin's own ``test.py`` covers that where the data is.
"""
import contextlib
import inspect
import socket
import warnings
from pathlib import Path

import pytest

import brainscore
from brainscore_core.plugin_management.parse_plugin_changes import get_plugin_ids
from brainscore_core.streaming import Session

pytestmark = pytest.mark.unit

PACKAGE = Path(brainscore.__file__).parent


def _unstated(registry, kind: str, root: Path) -> list:
    """The ``(folder, identifier)`` pairs of ``registry`` that the plugin runner cannot read in
    the plugin folder of ``root/kind`` that owns them (folder ``None``: a shared module)."""
    folders = sorted(init.parent.name for init in (root / kind).glob('*/__init__.py')
                     if not init.parent.name.startswith('_'))
    stated = {}
    for folder in folders:
        try:
            stated[folder] = get_plugin_ids(kind, [folder], str(root))
        except Exception:  # noqa: BLE001  the runner cannot read this folder at all
            stated[folder] = []
    everywhere = {identifier for identifiers in stated.values() for identifier in identifiers}
    unstated = []
    for identifier, factory in registry.items():
        source = Path(inspect.getsourcefile(getattr(factory, 'func', factory)) or '').resolve()
        if root.resolve() not in source.parents:
            continue                                    # a plugin of another package
        part = source.relative_to(root.resolve()).parts
        folder = part[1] if len(part) > 2 and part[0] == kind and part[1] in stated else None
        if identifier not in (stated[folder] if folder else everywhere):
            unstated.append((folder, identifier))
    return sorted(unstated, key=str)


@pytest.mark.parametrize('kind, registry', [('models', 'model_registry'),
                                            ('benchmarks', 'benchmark_registry')])
def test_every_identifier_is_stated_on_a_line_of_its_own_folder(kind, registry):
    unstated = _unstated(getattr(brainscore, registry), kind, PACKAGE)
    assert not unstated, (
        f'{unstated} (folder, identifier) are registered but core\'s plugin runner cannot read '
        f"them in the folder's __init__.py; write one line for each identifier: "
        f"{registry}['<identifier>'] = <factory>")


@pytest.mark.parametrize('lines, unstated', [
    ("benchmark_registry['a'] = _factory\n", []),
    ("benchmark_registry[\n    'a'] = _factory\n", [('one', 'a')]),          # unreadable line
    ("for key in ('a',):\n    benchmark_registry[key] = _factory\n", [('one', 'a')]),
])
def test_the_line_check_reads_the_lines_as_the_plugin_runner(lines, unstated, tmp_path):
    folder = tmp_path / 'benchmarks' / 'one'
    folder.mkdir(parents=True)
    init = folder / '__init__.py'
    init.write_text('def _factory():\n    pass\n' + lines)
    space = {}
    exec(compile(init.read_text().replace('benchmark_registry[', 'registry[') , str(init), 'exec'),
         {'registry': space})
    (tmp_path / 'benchmarks' / 'two').mkdir()
    (tmp_path / 'benchmarks' / 'two' / '__init__.py').write_text("benchmark_registry['b'] = f\n")
    space['b'] = space['a']                             # stated in another folder only
    space['external'] = lambda: None                    # a plugin of another package
    assert _unstated(space, 'benchmarks', tmp_path) == sorted(unstated + [('one', 'b')], key=str)


class _SessionStarted(Exception):
    pass


class _NetworkBlocked(OSError):
    pass


def _absent(error) -> bool:
    """Whether ``error``, an error it came from or an error it holds (in its arguments or its
    attributes: a fetcher that tries several sources raises one error with the list of their
    errors, each holding its connection error) says that data or an environment is absent."""
    pending, seen = [error], set()
    while pending:
        error = pending.pop()
        if not isinstance(error, BaseException) or id(error) in seen:
            continue
        seen.add(id(error))
        if isinstance(error, (brainscore.LoadError, FileNotFoundError, _NetworkBlocked)):
            return True
        values = [*error.args, *vars(error).values()]
        held = [item for value in values
                for item in (value.values() if isinstance(value, dict)
                             else value if isinstance(value, (list, tuple)) else [value])]
        pending += [error.__cause__, error.__context__, *held]
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


def _download_from_two_sources():
    errors = []
    for _ in range(2):
        try:
            socket.socket().connect(('127.0.0.1', 9))
        except OSError as error:
            errors.append(RuntimeError('could not connect'))
            errors[-1].kwargs = {'error': error}           # held, not chained
    raise Exception(errors)


@pytest.mark.parametrize('make, fails', [
    (lambda asset: _absent_data, False),
    (lambda asset: _download_from_two_sources, False),
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
