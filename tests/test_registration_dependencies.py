"""The two UMI vision registrations declare the unpublished packages they import.

Vision's plugin runner installs each changed plugin into a fresh environment
holding only the candidate vision checkout and its pinned core. Each UMI
registration therefore needs its own ``requirements.txt`` for ``brainscore``
and that package's unpublished ``brainscore-language`` dependency.

Everything here reads files and distribution metadata. pip, the network and the
model frameworks are never touched; ``brainscore_vision`` is located, not imported.
"""
import ast
import importlib.metadata as metadata
from importlib.util import find_spec
from pathlib import Path
import re
import subprocess
import sys
import types

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest

from brainscore.validation.revisions import REPOSITORIES
from brainscore_core.plugin_management import import_plugin as plugin_installer
from brainscore_core.plugin_management.parse_plugin_changes import get_plugin_ids, parse_plugin_changes

pytestmark = pytest.mark.unit

FOLDERS = {'clip-vit-b-32': 'clip_vit_b_32_unified', 'qwen2.5-vl-3b': 'qwen25_vl_3b_unified'}
HOST = 'brainscore-vision'  # the distribution whose plugin runner installs these files
SOURCE = {'brainscore': 'unified', 'brainscore-language': 'language'}  # distribution -> REPOSITORIES key


def _vision_root() -> Path:
    spec = find_spec('brainscore_vision')  # a top-level lookup does not import the package
    return Path(list(spec.submodule_search_locations)[0])


def _requirements(folder) -> list:
    path = _vision_root() / 'models' / folder / 'requirements.txt'
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [Requirement(line) for line in lines if line and not line.startswith('#')]


def _brainscore_family(distribution) -> set:
    return {canonicalize_name(Requirement(r).name) for r in metadata.requires(distribution) or []
            if canonicalize_name(Requirement(r).name).startswith('brainscore')}


def _imported_packages(folder) -> set:
    tree = ast.parse((_vision_root() / 'models' / folder / 'model.py').read_text())
    names = set()
    for node in ast.walk(tree):  # Qwen imports the wrappers inside its factory
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split('.')[0])
    return names


@pytest.mark.parametrize('identifier', FOLDERS)
def test_standard_installer_pip_installs_the_plugin_requirements(identifier, monkeypatch):
    root = _vision_root()
    # ImportPlugin only needs the package location; a stand-in avoids importing vision.
    monkeypatch.setitem(sys.modules, 'brainscore_vision',
                        types.SimpleNamespace(__file__=str(root / '__init__.py')))
    calls = []
    copied = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        copied.append(Path(command[-1]).read_text())        # the requirements copy pip is given
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'yes')
    monkeypatch.setattr(plugin_installer.subprocess, 'run', run)

    importer = plugin_installer.ImportPlugin('brainscore_vision', 'models', identifier)
    importer.install_requirements()

    requirements = root / 'models' / FOLDERS[identifier] / 'requirements.txt'
    assert importer.plugin_dirname == FOLDERS[identifier]
    (command, kwargs), = calls
    assert command[:5] == [sys.executable, '-m', 'pip', 'install', '-c'] and command[-2] == '-r'
    assert kwargs == {'capture_output': True, 'text': True}
    assert copied[0].split() == requirements.read_text().split()


@pytest.mark.parametrize('folder', FOLDERS.values())
def test_requirements_cover_brainscore_and_its_unpublished_language_peer(folder):
    assert 'brainscore' in _imported_packages(folder)
    host = {canonicalize_name(HOST)} | _brainscore_family(HOST)
    needed = ({'brainscore'} | _brainscore_family('brainscore')) - host
    declared = {canonicalize_name(requirement.name) for requirement in _requirements(folder)}
    assert 'brainscore-language' in needed
    assert declared == needed
    assert not declared & host, 'a core or vision URL would replace the candidate under test'


@pytest.mark.parametrize('folder', FOLDERS.values())
def test_each_declaration_is_an_immutable_source_revision(folder):
    for requirement in _requirements(folder):
        repository = REPOSITORIES[SOURCE[canonicalize_name(requirement.name)]]
        assert re.fullmatch(rf'git\+https://github\.com/{re.escape(repository)}\.git@[0-9a-f]{{40}}',
                            requirement.url), requirement
        assert not requirement.specifier and not requirement.marker


def test_both_registrations_list_the_same_pins():
    clip, qwen = (sorted(map(str, _requirements(folder))) for folder in FOLDERS.values())
    assert clip == qwen


def test_requirement_files_select_only_their_two_plugins():
    changed = ' '.join(f'brainscore_vision/models/{folder}/requirements.txt' for folder in FOLDERS.values())
    info = parse_plugin_changes(changed, 'brainscore_vision')
    assert sorted(info['changed_plugins']['models']) == sorted(FOLDERS.values())
    assert all(not info['changed_plugins'][kind] for kind in ('benchmarks', 'data', 'metrics'))
    assert info['test_all_plugins'] == []
    assert info['modifies_plugins'] and info['is_automergeable']
    assert sorted(get_plugin_ids('models', list(FOLDERS.values()), str(_vision_root()))) == sorted(FOLDERS)


def test_a_file_directly_under_models_would_select_every_model():
    # Why the pins live in each plugin rather than one shared file beside them.
    info = parse_plugin_changes('brainscore_vision/models/unified_requirements.txt', 'brainscore_vision')
    assert info['test_all_plugins'] == ['models']
