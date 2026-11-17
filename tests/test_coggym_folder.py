"""Brain-Score makes CogGym's folder (the pinned checkout and CogGym's own environment) at the
first use, completely or not at all, and never changes the scoring environment.

git, venv and pip are stand-ins that record their commands: nothing is cloned, made or installed.
"""
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from brainscore import LoadError
from brainscore.harnesses import _coggym_folder
from brainscore.harnesses._coggym_folder import ready_folder

from .coggym_stand_in import write_checkout

pytestmark = pytest.mark.unit

BENCHMARK = 'CogGym.Hu2023Fine.exp1'
# The commit the CogGym integration was checked against, written here on its own.
PINNED_COMMIT = 'a1cd9df1118fec80eba7463de237d1497d77e041'


@pytest.fixture
def tools(monkeypatch, tmp_path):
    """Stand-ins for git, venv and pip; ``tools.fail`` makes the command that contains it fail."""
    calls, lock = [], threading.Lock()
    state = type('Tools', (), {'calls': calls, 'fail': None, 'head': PINNED_COMMIT})()
    real = subprocess.run

    def run(command, *args, **kwargs):
        if command[0] != 'git' and command[1:3] != ['-m', 'venv'] and 'pip' not in command:
            return real(command, *args, **kwargs)
        with lock:
            calls.append(list(command))
        if state.fail and state.fail in command:
            raise subprocess.CalledProcessError(1, command, '', f'{state.fail} failed')
        stdout = ''
        if command[0] == 'git' and 'clone' in command:
            write_checkout(command[-1])
        elif command[0] == 'git' and 'rev-parse' in command:
            stdout = state.head + '\n'
        elif command[1:3] == ['-m', 'venv']:       # the environment's Python, where the folder expects it
            python = _coggym_folder.environment_python(Path(command[-1]).parent)
            python.parent.mkdir(parents=True)
            python.write_text('#!/bin/sh\n')
        elif 'freeze' in command:
            stdout = 'google-genai==3.2.0\n'
        return subprocess.CompletedProcess(command, 0, stdout, '')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setenv('BRAINSCORE_COGGYM_FOLDER', str(tmp_path / 'root'))
    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'yes')
    return state


def test_the_requirements_pin_every_package():
    lines = [line.split('#')[0].strip() for line in _coggym_folder.REQUIREMENTS.read_text().splitlines()]
    packages = [line for line in lines if line]
    assert packages
    floating = [line for line in packages if not re.fullmatch(r'[A-Za-z0-9_.\-]+==[^\s=*,<>;@]+', line)]
    assert floating == [], f'every package needs an exact version: {floating}'


def test_a_fresh_machine_gets_the_folder_at_the_pinned_commit_with_its_own_environment(tools, tmp_path):
    folder = ready_folder(BENCHMARK)
    assert folder.parent == tmp_path / 'root' and PINNED_COMMIT in folder.name
    marker = json.loads((folder / 'ready.json').read_text())
    assert marker['revision'] == PINNED_COMMIT and 'google-genai==3.2.0' in marker['freeze']
    clone = next(call for call in tools.calls if 'clone' in call)
    assert 'https://github.com/lance-ying/coggym.git' in clone
    assert any('checkout' in call and call[-1] == PINNED_COMMIT for call in tools.calls)
    install = next(call for call in tools.calls if 'install' in call)
    assert install[0].startswith(str(tmp_path / 'root'))       # the folder's own Python, not this one
    assert '--no-deps' in install and str(_coggym_folder.REQUIREMENTS) in install
    assert any('check' in call and 'pip' in call for call in tools.calls)
    assert sorted(path.name for path in folder.parent.iterdir()) == [folder.name]   # no partial folder
    # A second use finds the folder ready and runs nothing.
    count = len(tools.calls)
    assert ready_folder(BENCHMARK) == folder and len(tools.calls) == count


def test_with_installs_off_a_missing_folder_is_one_clear_error_and_nothing_runs(tools, tmp_path, monkeypatch):
    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'no')
    with pytest.raises(LoadError) as missing:
        ready_folder(BENCHMARK)
    message = str(missing.value)
    assert BENCHMARK in message and str(tmp_path / 'root') in message and 'BS_INSTALL_DEPENDENCIES' in message
    assert tools.calls == [] and not (tmp_path / 'root').exists()


@pytest.mark.parametrize('failing', ['clone', 'install', 'check'])
def test_a_failed_creation_is_a_clear_error_and_leaves_nothing(tools, tmp_path, failing):
    tools.fail = failing
    with pytest.raises(LoadError) as failed:
        ready_folder(BENCHMARK)
    assert BENCHMARK in str(failed.value) and 'Nothing was left' in str(failed.value)
    assert list((tmp_path / 'root').iterdir()) == []


def test_a_clone_at_another_commit_is_refused_and_leaves_nothing(tools, tmp_path):
    tools.head = 'f' * 40
    with pytest.raises(LoadError, match=PINNED_COMMIT):
        ready_folder(BENCHMARK)
    assert list((tmp_path / 'root').iterdir()) == []


def test_a_folder_the_user_had_is_never_changed(tools, tmp_path):
    folder = tmp_path / 'mine'
    folder.mkdir()
    (folder / 'notes.txt').write_text('my work')
    with pytest.raises(LoadError, match='not a ready CogGym folder'):
        ready_folder(BENCHMARK, folder)
    assert [path.name for path in folder.iterdir()] == ['notes.txt'] and tools.calls == []


def test_two_first_uses_at_once_make_the_folder_once(tools):
    folders, errors = [], []

    def use():
        try:
            folders.append(ready_folder(BENCHMARK))
        except Exception as error:      # pragma: no cover - reported below
            errors.append(error)
    threads = [threading.Thread(target=use) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and len(set(folders)) == 1
    assert sum('clone' in call for call in tools.calls) == 1
