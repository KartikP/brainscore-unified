"""The pin check must fail when the lock or a Jenkins script names a commit it must not."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'install' / 'check_pins.py'
CORE = 'a' * 40


@pytest.fixture
def pins(monkeypatch):
    spec = importlib.util.spec_from_file_location('check_pins', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through sys.modules.
    monkeypatch.setitem(sys.modules, 'check_pins', module)
    spec.loader.exec_module(module)
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
    monkeypatch.setattr(module, 'github_file', lambda repository, ref, path, api=False: None)
    _serve_lock_branches(module, monkeypatch, set())
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _commit(checkout: Path, message: str) -> str:
    """Commit everything in ``checkout`` (made a repository on first use); return the commit."""
    git = ['git', '-C', str(checkout), '-c', 'user.name=test', '-c', 'user.email=test@example.org']
    if not (checkout / '.git').exists():
        subprocess.run([*git, 'init', '-q'], check=True)
    subprocess.run([*git, 'add', '-A'], check=True)
    subprocess.run([*git, 'commit', '-q', '-m', message], check=True)
    return subprocess.run([*git, 'rev-parse', 'HEAD'], check=True, capture_output=True,
                          text=True).stdout.strip()


def _repositories(root: Path, *, core: str = CORE) -> Path:
    for name in ('core', 'vision', 'language'):
        _write(root / name / 'README.md', name)
    peers = {name: _commit(root / name, 'code') for name in ('vision', 'language')}
    _write(root / 'unified/install/peer-revisions.json', json.dumps({'core': core, **peers}))
    return root


def _infrastructure(root: Path, scripts: dict) -> Path:
    for name, line in scripts.items():
        _write(root / 'infra' / f'ci/{name}/run_tests.sh', f'    {line}\n')
    return root / 'infra'


FOLLOWS = {name: 'unified_ref=unified-model-interface-v2' for name in ('core', 'vision', 'language')}


def _run(module, root: Path, infrastructure: Path = None) -> int:
    arguments = []
    for name in ('core', 'vision', 'language', 'unified'):
        arguments += [f'--{name}', str(root / name)]
    infrastructure = infrastructure or _infrastructure(root, FOLLOWS)
    return module.main([*arguments, '--infrastructure', str(infrastructure)])


def test_lock_on_the_release_branches_passes(pins, tmp_path, capsys):
    assert _run(pins, _repositories(tmp_path)) == 0
    assert 'PASS' in capsys.readouterr().out


def test_lock_entry_off_the_release_branch_fails(pins, tmp_path, monkeypatch, capsys):
    root = _repositories(tmp_path)
    _serve_lock_branches(pins, monkeypatch, {CORE})
    assert _run(pins, root) == 1
    out = capsys.readouterr().out
    assert 'core lock entry on the release branch' in out and 'MISMATCH' in out


@pytest.mark.parametrize('core', [CORE[:7], 'unified-model-interface-v2', CORE + '-other'])
def test_lock_entry_must_be_a_full_commit(pins, tmp_path, core):
    assert _run(pins, _repositories(tmp_path, core=core)) == 1


def test_jenkins_script_that_pins_a_commit_fails(pins, tmp_path, capsys):
    root = _repositories(tmp_path)
    scripts = {**FOLLOWS, 'vision': 'unified_revision=' + 'b' * 40}
    assert _run(pins, root, _infrastructure(root, scripts)) == 1
    assert 'infrastructure ci/vision/run_tests.sh' in capsys.readouterr().out


def test_unreadable_jenkins_script_is_a_warning(pins, tmp_path, capsys):
    root = _repositories(tmp_path)
    assert _run(pins, root, _infrastructure(root, {'core': FOLLOWS['core']})) == 0
    assert 'warning: unreadable' in capsys.readouterr().out


def test_peer_pin_files_are_not_pin_sites(pins, tmp_path):
    """The caller workflows and requirements follow the release branches; a value there is not read."""
    root = _repositories(tmp_path)
    _write(root / 'vision/requirements-v2.txt', 'brainscore-core @ git+https://github.com/'
           'brain-score/core.git@' + 'e' * 40 + '\n')
    _write(root / 'core/.github/workflows/umi-v2.yml', 'uses: x/y@' + 'f' * 40 + '\n')
    assert _run(pins, root) == 0


def test_peer_manifest_must_name_the_code_of_the_peer_source(pins, tmp_path, capsys):
    """A later commit that moves only pin files keeps the named commit's code; a code change does not."""
    root = _repositories(tmp_path)
    _write(root / 'vision/requirements-v2.txt', 'brainscore-core @ git+https://github.com/'
           'brain-score/core.git@unified-model-interface-v2\n')
    _commit(root / 'vision', 'move the core requirement')
    assert _run(pins, root) == 0
    _write(root / 'vision/brainscore_vision/model.py', 'changed = True\n')
    _commit(root / 'vision', 'change code')
    assert _run(pins, root) == 1
    out = capsys.readouterr().out
    assert 'vision source' in out and 'MISMATCH' in out


def test_github_read_sends_the_token(monkeypatch):
    spec = importlib.util.spec_from_file_location('check_pins', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'check_pins', module)
    spec.loader.exec_module(module)
    monkeypatch.setenv('GH_TOKEN', 'secret')
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'text'

    def urlopen(request, timeout):
        seen['url'] = request.full_url
        seen['authorization'] = request.get_header('Authorization')
        return Response()

    monkeypatch.setattr(module, 'urlopen', urlopen)
    assert module.github_file('brain-score/core', 'v2', 'requirements-v2.txt') == 'text'
    assert seen['authorization'] == 'Bearer secret'
    assert seen['url'].endswith('/repos/brain-score/core/contents/requirements-v2.txt?ref=v2')


def _serve_trees(module, monkeypatch, trees: dict) -> None:
    """Serve ``git/trees/<ref>`` from ``trees``: ref -> {name: (type, sha)}."""
    def github_file(repository, ref, path, api=False):
        entries = trees.get(path.removeprefix('git/trees/'))
        if entries is None:
            return None
        return json.dumps({'tree': [{'path': name, 'type': kind, 'sha': sha}
                                    for name, (kind, sha) in entries.items()]})
    monkeypatch.setattr(module, 'github_file', github_file)


def test_squash_merged_source_with_equal_code_passes(pins, monkeypatch):
    """A squash merge has other history but the named commit's tree: no path differs."""
    trees = {'named': {'pkg': ('tree', 'P1'), 'README.md': ('blob', 'R1')},
             'branch': {'pkg': ('tree', 'P1'), 'README.md': ('blob', 'R1')}}
    _serve_trees(pins, monkeypatch, trees)
    source = pins.Source('vision', 'brain-score/vision', ref='branch')
    assert source.changed_since('named') == set()


def test_source_with_a_changed_file_names_the_file(pins, monkeypatch):
    trees = {'named': {'pkg': ('tree', 'P1'), 'README.md': ('blob', 'R1')},
             'branch': {'pkg': ('tree', 'P2'), 'README.md': ('blob', 'R1')},
             'P1': {'model.py': ('blob', 'M1'), 'data.py': ('blob', 'D1')},
             'P2': {'model.py': ('blob', 'M2'), 'data.py': ('blob', 'D1')}}
    _serve_trees(pins, monkeypatch, trees)
    source = pins.Source('vision', 'brain-score/vision', ref='branch')
    changed = source.changed_since('named')
    assert 'pkg/model.py' in changed and 'pkg/data.py' not in changed


def test_unreadable_tree_is_unknown(pins, monkeypatch):
    _serve_trees(pins, monkeypatch, {'branch': {}})
    assert pins.Source('vision', 'brain-score/vision', ref='branch').changed_since('named') is None


def _serve_lock_branches(module, monkeypatch, off_branch: set) -> None:
    """Serve the compare endpoint: a commit in ``off_branch`` is not on the release branch."""
    plain = module.github_file

    def github_file(repository, ref, path, api=False):
        if api and path.startswith('compare/'):
            commit = path.removeprefix('compare/').split('...')[0]
            return json.dumps({'status': 'diverged' if commit in off_branch else 'ahead'})
        return plain(repository, ref, path, api)
    monkeypatch.setattr(module, 'github_file', github_file)


def _serve_lock_branches(module, monkeypatch, off_branch: set) -> None:
    """Serve the compare endpoint: a commit in ``off_branch`` is not on the release branch."""
    plain = module.github_file

    def github_file(repository, ref, path, api=False):
        if api and path.startswith('compare/'):
            commit = path.removeprefix('compare/').split('...')[0]
            return json.dumps({'status': 'diverged' if commit in off_branch else 'ahead'})
        return plain(repository, ref, path, api)
    monkeypatch.setattr(module, 'github_file', github_file)
