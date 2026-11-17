"""The first load of a CogGym benchmark makes CogGym's folder; the scoring environment is never
written to.

git and the Python that makes the environment and runs pip are recording stand-ins
(``tests/coggym_commands.py``): nothing is cloned or installed.
"""
import os
from pathlib import Path

import pytest

import brainscore
from brainscore import LoadError

from .coggym_commands import PINNED_COMMIT, commands  # noqa: F401  (commands is a fixture)

pytestmark = pytest.mark.unit

BENCHMARK = 'CogGym.Hu2023Fine.exp1'


@pytest.fixture
def fresh(commands, tmp_path, monkeypatch):
    """Installs allowed and an empty place for CogGym's folders."""
    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'yes')
    monkeypatch.setenv('BRAINSCORE_COGGYM_FOLDER', str(tmp_path / 'folders'))
    return commands


def _load():
    return brainscore.load_benchmark(BENCHMARK, reset=lambda repetition: None)


def test_pip_cannot_be_sent_outside_the_folder_environment(fresh, tmp_path, monkeypatch):
    """Settings of this process that move pip's destination never reach the install."""
    site = tmp_path / 'scoring-site-packages'
    configuration = tmp_path / 'pip.conf'
    configuration.write_text(f'[install]\ntarget = {site}\n')
    monkeypatch.setenv('PIP_TARGET', str(site))
    monkeypatch.setenv('PIP_PREFIX', str(site))
    monkeypatch.setenv('PIP_ROOT', str(site))
    monkeypatch.setenv('PIP_USER', '1')
    monkeypatch.setenv('PIP_CONFIG_FILE', str(configuration))
    monkeypatch.setenv('GIT_DIR', str(tmp_path / 'some-repository'))
    _load()
    pip = fresh.of('pip')
    assert pip and all(str(tmp_path / 'folders') in call['executable'] for call in pip)
    for call in pip:
        assert not {'PIP_TARGET', 'PIP_PREFIX', 'PIP_ROOT', 'PIP_USER'} & set(call['env'])
        assert call['env']['PIP_CONFIG_FILE'] == os.devnull
    assert any('--require-virtualenv' in call['args'] for call in pip if 'install' in call['args'])
    assert all('GIT_DIR' not in call['env'] for call in fresh.of('git'))
    assert not site.exists()


def _default_folder(root):
    """The default folder name: the pinned commit and the first 12 characters of the hash of the
    shipped requirements."""
    import hashlib
    import brainscore.harnesses
    requirements = Path(brainscore.harnesses.__file__).with_name('coggym_requirements.txt')
    return root / f'{PINNED_COMMIT}-{hashlib.sha256(requirements.read_bytes()).hexdigest()[:12]}'


def test_a_folder_that_appears_while_the_folder_is_made_is_never_replaced(fresh, tmp_path):
    """Another program makes the destination (empty) while Brain-Score makes its folder."""
    destination = _default_folder(tmp_path / 'folders')
    fresh.behave(make=str(destination))
    with pytest.raises(LoadError) as refused:
        _load()
    assert str(destination) in str(refused.value)
    assert destination.is_dir() and list(destination.iterdir()) == []      # untouched
    assert not list(destination.parent.glob('.*.partial-*'))      # no temporary folder left


def test_a_folder_made_at_the_instant_of_publication_is_never_replaced(fresh, tmp_path, monkeypatch):
    """Another program makes the destination at the very moment Brain-Score publishes its folder.

    The other program acts inside the file-system call that publishes (``os.rename``, or the
    encoding of the destination's name for the atomic call): the narrowest race there is.
    """
    destination = _default_folder(tmp_path / 'folders')
    rename, fsencode = os.rename, os.fsencode
    made = []

    def other_program_then(call):
        def act(*args):
            if not destination.exists() and any(str(arg) == str(destination) for arg in args):
                destination.mkdir()             # an empty folder: the case a plain rename replaces
                made.append(destination.stat().st_ino)
            return call(*args)
        return act
    monkeypatch.setattr(os, 'rename', other_program_then(rename))
    monkeypatch.setattr(os, 'fsencode', other_program_then(fsencode))
    with pytest.raises(LoadError):
        _load()
    assert made and destination.stat().st_ino == made[0] and list(destination.iterdir()) == []
    assert not list(destination.parent.glob('.*.partial-*'))
