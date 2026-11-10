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
        printed = dict(line.split('=', 1) for line in result.stdout.splitlines()
                       if '=' in line)
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


def test_work_entry_not_created_by_the_script_is_never_removed(tmp_path):
    work = tmp_path / 'work'
    entry = _checkout(work / 'core')
    (work / '.umi-candidate').touch()
    result = _run('--unified', str(_checkout(tmp_path / 'unified-src')),
                  '--core', str(_checkout(tmp_path / 'src')), 'prepare', work=work)
    assert result.returncode != 0
    assert 'did not create it' in result.stderr
    assert (entry / 'keep.txt').read_text() == 'user data'


def test_replacing_a_link_keeps_its_target(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    (work / '.umi-candidate').touch()
    old = _checkout(tmp_path / 'old')
    (work / 'core').symlink_to(old)
    source = _checkout(tmp_path / 'src')
    others = [_checkout(tmp_path / name) for name in ('vision-src', 'language-src')]
    # The sources are not Git checkouts, so prepare stops after linking them.
    _run('--unified', str(_checkout(tmp_path / 'unified-src')), '--core', str(source),
         '--vision', str(others[0]), '--language', str(others[1]), 'prepare', work=work)
    assert (old / 'keep.txt').read_text() == 'user data'
    assert (work / 'core').resolve() == source.resolve()


def test_failed_phase_skips_later_phases_except_records(tmp_path):
    result = _run('openpi-client', work=tmp_path / 'work')
    assert result.returncode == 1
    output = result.stdout
    assert 'FAIL: openpi-client/verify-core' in result.stderr
    assert '==> openpi-client-install skipped after a failure' in output
    assert '==> openpi-client-record\n' in output
    assert '==> openpi-client-test skipped after a failure' in output
