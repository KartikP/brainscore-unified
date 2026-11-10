"""The shared CI runner must protect local paths and behave like the workflow."""

import os
from pathlib import Path
import shutil
import subprocess

import yaml  # installed with transformers


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'install' / 'run_candidate.sh'
WORKFLOW = ROOT / '.github' / 'workflows' / 'offline-tests.yml'


def _run(*arguments, work: Path):
    env = {'PATH': os.environ['PATH'], 'HOME': str(work.parent)}
    return subprocess.run([shutil.which('bash'), str(SCRIPT), '--work', str(work), *arguments],
                          capture_output=True, text=True, env=env, timeout=120)


def _checkout(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / 'keep.txt').write_text('user data')
    return path


def test_job_environment_matches_the_workflow(tmp_path):
    jobs = yaml.safe_load(WORKFLOW.read_text())['jobs']
    for job in ('offline', 'openpi-client', 'openpi-instrumentation'):
        result = _run('--job', job, 'print-env', work=tmp_path / 'work')
        assert result.returncode == 0, result.stderr
        # Skip the "==> print-env" phase header.
        printed = dict(line.split('=', 1) for line in result.stdout.splitlines()
                       if '=' in line and not line.startswith('==>'))
        assert printed == {key: str(value) for key, value in jobs[job]['env'].items()}, job


def test_local_path_inside_work_is_refused(tmp_path):
    work = tmp_path / 'work'
    user = _checkout(work / 'my-core')
    result = _run('--unified', str(_checkout(tmp_path / 'unified-src')), '--core', str(user),
                  'prepare', work=work)
    assert result.returncode != 0
    assert 'inside --work' in result.stderr
    assert (user / 'keep.txt').read_text() == 'user data'


def test_work_inside_local_path_is_refused(tmp_path):
    user = _checkout(tmp_path / 'core')
    result = _run('--unified', str(_checkout(tmp_path / 'unified-src')), '--core', str(user),
                  'prepare', work=user / 'work')
    assert result.returncode != 0
    assert 'is inside local path' in result.stderr
    assert (user / 'keep.txt').read_text() == 'user data'
    assert not (user / 'work').exists()


def _repository(path: Path) -> Path:
    """A local checkout with a commit, an uncommitted change and an untracked file."""
    _checkout(path)
    git = ['git', '-C', str(path), '-c', 'user.name=Test', '-c', 'user.email=test@example.com']
    subprocess.run(['git', 'init', '-q', str(path)], check=True)
    subprocess.run([*git, 'add', '-A'], check=True)
    subprocess.run([*git, 'commit', '-q', '-m', 'initial'], check=True)
    (path / 'keep.txt').write_text('uncommitted change')
    (path / 'untracked.txt').write_text('untracked')
    return path


def _listing(root: Path) -> dict:
    """Every file and folder below root, with type, size, mtime and content."""
    entries = {}
    for directory, folders, files in os.walk(root):
        for name in [*folders, *files]:
            path = Path(directory) / name
            info = path.lstat()
            content = path.read_bytes() if path.is_file() and not path.is_symlink() else None
            entries[str(path.relative_to(root))] = (info.st_mode, info.st_size,
                                                    info.st_mtime_ns, content)
    return entries


def test_dot_components_in_work_are_refused(tmp_path):
    work = tmp_path / 'work' / 'new' / '..'
    result = _run('--unified', str(_checkout(tmp_path / 'unified-src')), 'prepare', work=work)
    assert result.returncode != 0
    assert "must not contain '.' or '..'" in result.stderr
    assert not (tmp_path / 'work').exists()


def test_venv_inside_a_local_path_is_refused(tmp_path):
    user = _checkout(tmp_path / 'core')
    result = _run('--unified', str(_checkout(tmp_path / 'unified-src')), '--core', str(user),
                  '--venv', str(user / 'venv'), 'prepare', work=tmp_path / 'work')
    assert result.returncode != 0
    assert 'overlaps local path' in result.stderr
    assert not (user / 'venv').exists()


def test_venv_outside_work_is_refused(tmp_path):
    result = _run('--unified', str(_checkout(tmp_path / 'unified-src')),
                  '--venv', str(tmp_path / 'venvs'), 'prepare', work=tmp_path / 'work')
    assert result.returncode != 0
    assert 'must be a folder inside --work' in result.stderr


def test_work_entry_not_created_by_the_script_is_never_removed(tmp_path):
    work = tmp_path / 'work'
    entry = _checkout(work / 'core')
    (work / '.umi-candidate').touch()
    result = _run('--unified', str(_repository(tmp_path / 'unified-src')),
                  '--core', str(_repository(tmp_path / 'src')), 'prepare', work=work)
    assert result.returncode != 0
    assert 'did not create it' in result.stderr
    assert (entry / 'keep.txt').read_text() == 'user data'


def test_replacing_a_link_keeps_its_target(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    (work / '.umi-candidate').touch()
    old = _checkout(tmp_path / 'old')
    (work / 'core').symlink_to(old)
    # vision is not a Git checkout, so prepare stops after the core snapshot.
    _run('--unified', str(_repository(tmp_path / 'unified-src')),
         '--core', str(_repository(tmp_path / 'src')),
         '--vision', str(_checkout(tmp_path / 'vision-src')), 'prepare', work=work)
    assert _listing(old) == {'keep.txt': _listing(old)['keep.txt']}
    assert (old / 'keep.txt').read_text() == 'user data'
    assert not (work / 'core').is_symlink()
    assert (work / 'core' / '.git' / 'umi-managed').is_file()


def test_local_checkouts_are_unchanged_and_snapshotted(tmp_path):
    sources = {name: _repository(tmp_path / f'{name}-src')
               for name in ('unified', 'core', 'vision', 'language')}
    before = {name: _listing(path) for name, path in sources.items()}
    work = tmp_path / 'work'
    # The fake unified has no revisions.py, so prepare stops after the snapshots.
    result = _run(*[part for name, path in sources.items() for part in (f'--{name}', str(path))],
                  '--venv', str(work / 'venvs'), 'prepare', work=work)
    assert result.returncode != 0
    assert {name: _listing(path) for name, path in sources.items()} == before
    for name, source in sources.items():
        snapshot = work / name
        assert not snapshot.is_symlink()
        assert (snapshot / 'keep.txt').read_text() == 'uncommitted change'
        assert (snapshot / 'untracked.txt').read_text() == 'untracked'
        heads = [subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], check=True,
                                capture_output=True, text=True).stdout
                 for path in (source, snapshot)]
        assert heads[0] == heads[1]


def test_failed_phase_skips_later_phases_except_records(tmp_path):
    result = _run('openpi-client', work=tmp_path / 'work')
    assert result.returncode == 1
    output = result.stdout
    assert 'FAIL: openpi-client/verify-core' in result.stderr
    assert '==> openpi-client-install skipped after a failure' in output
    assert '==> openpi-client-record\n' in output
    assert '==> openpi-client-test skipped after a failure' in output
