"""The first load of a CogGym benchmark makes CogGym's folder; the scoring environment is never
written to.

git and the Python that makes the environment and runs pip are recording stand-ins
(``tests/coggym_commands.py``): nothing is cloned or installed.
"""
import os

import pytest

import brainscore

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
