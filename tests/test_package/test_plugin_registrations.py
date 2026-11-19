"""Two checks of the model and benchmark plugins. Each makes one claim; it names no plugin, so
a new plugin is checked when its folder exists.

- Each registered identifier is on a literal line. Every identifier that this package
  registers (the model and benchmark registries of a fresh process that imports only
  ``brainscore``, so a registration that a test or another package adds is not judged) is
  found by core's static reader (``get_plugin_ids``, the reader of the plugin runner) in the
  ``__init__.py`` of some plugin folder of the same kind, and every identifier that the reader
  finds in a folder that imported is registered. A key made in a loop or a function is not
  found and fails. Not covered: which folder owns an identifier (a folder that registers a
  computed key that another folder writes as a literal passes), and the folders whose import
  failed (they register nothing).
- Building a benchmark starts no session (Contract Rules 5.6: nothing runs before ``score()``
  can return N/A). A session constructed during the build fails the check, also when the
  build catches the refusal and also when the build then ends with an exception. A build that
  ends with an exception and constructed no session is not judged; the check counts these
  builds and states the count in a ``BuildsNotJudged`` warning. Not covered: a session that a
  plugin constructs when its module is imported, before the factory runs; loaders, downloads,
  child processes and installs during a build (Contract Rules 5.6 allows a loader before
  evaluation); and a build that fails (the plugin's own tests and core's generic plugin tests
  cover it).
"""
import functools
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import brainscore
from brainscore_core.plugin_management.parse_plugin_changes import get_plugin_ids
from brainscore_core.streaming import Session

pytestmark = pytest.mark.unit

PACKAGE = Path(brainscore.__file__).parent

_REGISTERED_BY_THE_PACKAGE = '''
import json, sys, brainscore
print(json.dumps({
    'models': sorted(brainscore.model_registry), 'benchmarks': sorted(brainscore.benchmark_registry),
    'imported': sorted(name for name in sys.modules
                       if name.startswith(('brainscore.models.', 'brainscore.benchmarks.')))}))
'''


@functools.lru_cache(maxsize=None)
def _registered_by_the_package() -> dict:
    """The identifiers that ``import brainscore`` alone registers, and the plugin modules it
    imported, read in a fresh process: the registries of this process can hold more."""
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), BS_INSTALL_DEPENDENCIES='no')
    result = subprocess.run([sys.executable, '-W', 'ignore', '-c', _REGISTERED_BY_THE_PACKAGE],
                            env=environment, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-600:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _unread(live, kind: str, root: Path, imported) -> tuple:
    """``(not on a literal line, on a line and not registered)``: the identifiers of ``live``
    that the static reader finds in no folder of ``root/kind``, and the identifiers that it
    finds in a folder named by ``imported(folder)`` that ``live`` lacks."""
    folders = sorted(init.parent.name for init in (root / kind).glob('*/__init__.py')
                     if not init.parent.name.startswith('_'))
    stated, loaded = set(), set()
    for folder in folders:
        ids = set(get_plugin_ids(kind, [folder], str(root)))
        stated |= ids
        if imported(folder):
            loaded |= ids
    return sorted(set(live) - stated), sorted(loaded - set(live))


@pytest.mark.parametrize('kind, registry', [('models', 'model_registry'),
                                            ('benchmarks', 'benchmark_registry')])
def test_every_identifier_is_on_a_literal_line(kind, registry, monkeypatch):
    # A registration from outside the package, as a test or a user's own package makes it.
    monkeypatch.setitem(getattr(brainscore, registry), 'a-key-from-outside-the-package', lambda: None)
    fresh = _registered_by_the_package()
    unread, unregistered = _unread(fresh[kind], kind, PACKAGE,
                                   lambda folder: f'brainscore.{kind}.{folder}' in fresh['imported'])
    assert not unread, (f'{unread}: registered, but on no line that the plugin runner reads; write '
                        f"one line in the plugin's __init__.py for each: {registry}['<identifier>'] = <factory>")
    assert not unregistered, f'{unregistered}: on a line of an imported plugin and not registered'


@pytest.mark.parametrize('line, live, fails', [
    ("benchmark_registry['a'] = factory\n", {'a'}, False),
    ("for key in ('a',):\n    benchmark_registry[key] = factory\n", {'a'}, True),
])
def test_the_line_check_fails_for_a_key_made_in_a_loop(line, live, fails, tmp_path):
    (tmp_path / 'benchmarks' / 'one').mkdir(parents=True)
    (tmp_path / 'benchmarks' / 'one' / '__init__.py').write_text(
        'from brainscore import benchmark_registry\n' + line)
    unread, unregistered = _unread(live, 'benchmarks', tmp_path, lambda folder: True)
    assert bool(unread or unregistered) == fails


class _Refused(Exception):
    """Raised in place of a session that a build constructs."""


class BuildsNotJudged(Warning):
    """The count of builds that ended with an exception. Not a ``UserWarning``: the package's
    pytest settings hide those, and this count must show in the test output."""


def _judge_build(factory, monkeypatch) -> str:
    """``'session'`` when the build constructs a session (also when it catches the refusal,
    also when it then ends with an exception), ``'not judged'`` when it ends with an exception
    and constructed none, else ``'no session'``."""
    started = []

    def refuse(*args, **kwargs):
        started.append(True)
        raise _Refused('a session is refused while a benchmark is built')
    with monkeypatch.context() as patch:
        patch.setenv('BS_INSTALL_DEPENDENCIES', 'no')      # a build installs nothing on this machine
        patch.setattr(Session, '__init__', refuse)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                factory()
        except Exception:  # noqa: BLE001  any exception: the build did not end
            return 'session' if started else 'not judged'
    return 'session' if started else 'no session'


def test_building_a_benchmark_starts_no_session(monkeypatch):
    judged = {identifier: _judge_build(factory, monkeypatch)
              for identifier, factory in sorted(brainscore.benchmark_registry.items())}
    started = [identifier for identifier, outcome in judged.items() if outcome == 'session']
    assert not started, (f'building {started} starts a session; run the work in __call__ '
                         'or compute it at first use')
    not_judged = sum(outcome == 'not judged' for outcome in judged.values())
    warnings.warn(f'{not_judged} of {len(judged)} benchmark builds ended with an exception and '
                  'were not judged for the session rule', BuildsNotJudged)


def _session():
    Session(None, None, None)


def _session_caught():
    try:
        _session()
    except Exception:  # noqa: BLE001  a build that hides the refusal
        pass


def _raises():
    raise FileNotFoundError('an asset is absent')


def _session_caught_then_raises():
    _session_caught()
    _raises()


@pytest.mark.parametrize('build, outcome', [
    (lambda: None, 'no session'),
    (_session, 'session'),
    (_session_caught, 'session'),
    (_raises, 'not judged'),
    (_session_caught_then_raises, 'session'),
])
def test_the_build_check_sees_a_session_also_when_caught(build, outcome, monkeypatch):
    assert _judge_build(build, monkeypatch) == outcome
