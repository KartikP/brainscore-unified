"""Brain-Score makes CogGym's folder (the pinned checkout and CogGym's own environment) at the
first load of a CogGym benchmark, completely or not at all.

Every test loads a benchmark through ``brainscore.load_benchmark``. git and the Python that
makes the environment and runs pip are recording stand-ins (``tests/command_stand_ins.py``):
nothing is cloned, made or installed.
"""
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import brainscore
from brainscore import LoadError

from tests.command_stand_ins import PINNED_COMMIT, commands  # noqa: F401  (commands is a fixture)

pytestmark = pytest.mark.unit

BENCHMARK = 'CogGym.Hu2023Fine.exp1'
REQUIREMENTS = Path(brainscore.__file__).parent / 'harnesses' / 'coggym_requirements.txt'


@pytest.fixture
def fresh(commands, tmp_path, monkeypatch):
    """Installs allowed and an empty place for CogGym's folders."""
    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'yes')
    monkeypatch.setenv('BRAINSCORE_ENVIRONMENTS_HOME', str(tmp_path / 'folders'))
    return commands


def _load(**configuration):
    return brainscore.load_benchmark(BENCHMARK, reset=lambda repetition: None, **configuration)


def _partials(root):
    return list(Path(root).glob('.*.partial-*'))


def test_the_requirements_pin_every_package():
    lines = [line.split('#')[0].strip() for line in REQUIREMENTS.read_text().splitlines()]
    packages = [line for line in lines if line]
    assert packages
    floating = [line for line in packages if not re.fullmatch(r'[A-Za-z0-9_.\-]+==[^\s=*,<>;@]+', line)]
    assert floating == [], f'every package needs an exact version: {floating}'
    # CogGym's own requirements at the pinned commit are in the list.
    names = {line.split('==')[0].lower().replace('_', '-') for line in packages}
    assert {'google-genai', 'python-dotenv', 'pillow'} <= names


def test_a_fresh_machine_gets_the_folder_at_the_pinned_commit_with_its_own_environment(fresh, tmp_path):
    _load()
    (folder,) = [path for path in (tmp_path / 'folders' / 'CogGym').iterdir() if not path.name.startswith('.')]
    assert PINNED_COMMIT in folder.name
    marker = json.loads((folder / 'ready.json').read_text())
    assert marker['revision'] == PINNED_COMMIT and 'google-genai==3.2.0' in marker['freeze']
    assert marker['python'] == platform.python_version()       # this Python made the environment
    clone = next(call for call in fresh.of('git') if 'clone' in call['args'])
    assert 'https://github.com/lance-ying/coggym.git' in clone['args']
    assert any('checkout' in call['args'] and call['args'][-1] == PINNED_COMMIT for call in fresh.of('git'))
    install = next(call for call in fresh.of('pip') if 'install' in call['args'])
    assert install['executable'].startswith(str(tmp_path / 'folders'))     # the folder's Python, not this one
    assert '--no-deps' in install['args'] and str(REQUIREMENTS) in install['args']
    assert any('check' in call['args'] for call in fresh.of('pip'))
    assert _partials(tmp_path / 'folders' / 'CogGym') == []
    # A second load finds the folder ready: nothing is cloned or installed again.
    count = len(fresh.of('pip')) + len(fresh.of('venv'))
    _load()
    assert len(fresh.of('pip')) + len(fresh.of('venv')) == count
    assert sum('clone' in call['args'] for call in fresh.of('git')) == 1


def test_with_installs_off_a_missing_folder_is_one_clear_error_and_nothing_runs(fresh, tmp_path, monkeypatch):
    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'no')
    with pytest.raises(LoadError) as missing:
        _load()
    message = str(missing.value)
    assert BENCHMARK in message and str(tmp_path / 'folders') in message and 'BS_INSTALL_DEPENDENCIES' in message
    assert fresh.calls == [] and not (tmp_path / 'folders').exists()


@pytest.mark.parametrize('failing', ['clone', 'venv', 'install', 'check'])
def test_a_failed_creation_is_a_clear_error_and_leaves_no_folder(fresh, tmp_path, failing):
    fresh.behave(fail=failing)
    with pytest.raises(LoadError) as failed:
        _load()
    assert BENCHMARK in str(failed.value) and 'Nothing was left' in str(failed.value)
    assert [path for path in (tmp_path / 'folders' / 'CogGym').iterdir() if not path.name.endswith('.lock')] == []


def test_a_clone_at_another_commit_is_refused_and_leaves_no_folder(fresh, tmp_path):
    fresh.behave(head='f' * 40)
    with pytest.raises(LoadError, match=PINNED_COMMIT):
        _load()
    assert [path for path in (tmp_path / 'folders' / 'CogGym').iterdir() if not path.name.endswith('.lock')] == []


def test_a_folder_the_user_had_is_never_changed(fresh, tmp_path):
    folder = tmp_path / 'mine'
    folder.mkdir()
    (folder / 'notes.txt').write_text('my work')
    with pytest.raises(LoadError, match='not a ready CogGym folder'):
        _load(folder=folder)
    assert [path.name for path in folder.iterdir()] == ['notes.txt'] and fresh.calls == []


def test_two_first_loads_at_once_make_the_folder_once(fresh):
    errors = []

    def load():
        try:
            _load()
        except Exception as error:      # reported below
            errors.append(error)
    threads = [threading.Thread(target=load) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sum('clone' in call['args'] for call in fresh.of('git')) == 1


def test_a_killed_creator_leaves_no_command_running_and_the_next_load_succeeds(fresh, tmp_path):
    """The process that makes the folder is killed while git runs: git is stopped, and the next
    load removes the abandoned temporary folder and makes the folder."""
    fresh.behave(sleep='clone')
    creator = subprocess.Popen([sys.executable, '-c', (
        'import brainscore; '
        f'brainscore.load_benchmark({BENCHMARK!r}, reset=lambda repetition: None)')],
        env=dict(os.environ))
    pid_file = Path(str(fresh.log) + '.pid')
    deadline = time.monotonic() + 30
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    git = int(pid_file.read_text())
    creator.send_signal(signal.SIGKILL)
    creator.wait()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(git, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(git, signal.SIGKILL)
        pytest.fail('git outlived the process that started it')
    assert _partials(tmp_path / 'folders' / 'CogGym')            # the killed creator's temporary folder
    fresh.behave()
    _load()
    assert _partials(tmp_path / 'folders' / 'CogGym') == []


def test_a_read_only_place_is_one_clear_error(fresh, tmp_path, monkeypatch):
    read_only = tmp_path / 'read-only'
    read_only.mkdir()
    read_only.chmod(0o500)
    monkeypatch.setenv('BRAINSCORE_ENVIRONMENTS_HOME', str(read_only / 'nested'))
    try:
        with pytest.raises(LoadError) as refused:
            _load()
    finally:
        read_only.chmod(0o700)
    message = str(refused.value)
    assert BENCHMARK in message and str(read_only / 'nested') in message
    assert 'BRAINSCORE_ENVIRONMENTS_HOME' in message and fresh.calls == []


def test_a_sibling_that_only_looks_like_a_temporary_folder_is_never_removed(fresh, tmp_path):
    from tests.coggym_stand_in import default_place
    folder = default_place(tmp_path / 'folders')
    theirs = folder.parent / f'.{folder.name}.partial-user-not-a-creation'
    theirs.mkdir(parents=True)
    (theirs / 'my-research.txt').write_text('mine')
    _load()
    assert (theirs / 'my-research.txt').read_text() == 'mine'
    assert (folder / 'ready.json').exists()


def test_a_file_of_the_lock_name_keeps_its_content(fresh, tmp_path):
    from tests.coggym_stand_in import default_place
    folder = default_place(tmp_path / 'folders')
    folder.parent.mkdir(parents=True)
    theirs = folder.parent / f'.{folder.name}.lock'
    theirs.write_text('my notes')
    _load()
    assert theirs.read_text() == 'my notes' and (folder / 'ready.json').exists()


def test_a_link_at_the_lock_name_is_refused_and_its_target_kept(fresh, tmp_path):
    from tests.coggym_stand_in import default_place
    folder = default_place(tmp_path / 'folders')
    folder.parent.mkdir(parents=True)
    target = tmp_path / 'notes.txt'
    target.write_text('my notes')
    (folder.parent / f'.{folder.name}.lock').symlink_to(target)
    with pytest.raises(LoadError) as refused:
        _load()
    assert BENCHMARK in str(refused.value) and target.read_text() == 'my notes'
    assert not folder.exists()


def test_a_record_that_cannot_be_removed_after_publication_does_not_fail_the_load(fresh, tmp_path, monkeypatch):
    """The complete folder becomes read-only at the moment it is published: the load succeeds, the
    folder stays, and the next load uses it."""
    from tests.coggym_stand_in import default_place
    folder = default_place(tmp_path / 'folders')
    fsencode = os.fsencode

    def read_only_at_publication(name):
        text = os.fspath(name)
        if os.path.basename(text).startswith(f'.{folder.name}.partial-') and os.path.isdir(text):
            os.chmod(text, 0o500)
        return fsencode(name)
    monkeypatch.setattr(os, 'fsencode', read_only_at_publication)
    try:
        _load()
        monkeypatch.setattr(os, 'fsencode', fsencode)
        assert (folder / 'ready.json').exists()
        _load()                                        # the next load finds the folder ready
        assert (folder / 'ready.json').exists()
    finally:
        if folder.exists():
            folder.chmod(0o700)


def test_a_complete_folder_renamed_to_a_temporary_name_is_never_removed(fresh, tmp_path, monkeypatch):
    """A published folder that kept its creation record is renamed by the user to a name of the
    temporary pattern and holds the user's file: the next load leaves it in place."""
    from tests.coggym_stand_in import default_place
    folder = default_place(tmp_path / 'folders')
    fsencode = os.fsencode

    def read_only_at_publication(name):
        text = os.fspath(name)
        if os.path.basename(text).startswith(f'.{folder.name}.partial-') and os.path.isdir(text):
            os.chmod(text, 0o500)
        return fsencode(name)
    monkeypatch.setattr(os, 'fsencode', read_only_at_publication)
    _load()                                          # the record stays in the published folder
    monkeypatch.setattr(os, 'fsencode', fsencode)
    folder.chmod(0o700)
    saved = folder.parent / f'.{folder.name}.partial-saved'
    folder.rename(saved)
    (saved / 'saved-result.txt').write_text('mine')
    _load()
    assert (saved / 'saved-result.txt').read_text() == 'mine'
    assert (folder / 'ready.json').exists()


@pytest.mark.parametrize('value', ['', 'relative/folders'])
def test_a_place_that_is_empty_or_not_absolute_is_refused_before_anything_runs(fresh, monkeypatch, value):
    monkeypatch.setenv('BRAINSCORE_ENVIRONMENTS_HOME', value)
    with pytest.raises(LoadError) as refused:
        _load()
    assert BENCHMARK in str(refused.value) and 'BRAINSCORE_ENVIRONMENTS_HOME' in str(refused.value)
    assert fresh.calls == []
