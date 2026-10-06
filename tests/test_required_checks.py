"""Exercise pytest itself so skipped/empty runs cannot masquerade as success."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('source,code,passed,unexpected', [
    ('def test_ok(): assert True', 0, 1, 0),
    ('import pytest\ndef test_skip(): pytest.skip("data missing")', 1, 0, 1),
    ('import pytest\ndef test_ok(): pass\ndef test_skip(): pytest.skip("missing")', 1, 1, 1),
    ('import pytest\npytest.importorskip("missing_umi_test_dependency")', 5, 0, 1),
    ('def test_bad(): assert False', 1, 0, 0),
    ('', 5, 0, 0),
])
def test_required_profile_exit_status(tmp_path, source, code, passed, unexpected):
    (tmp_path / 'test_sample.py').write_text(source)
    report = tmp_path / 'checks.json'
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    paths = [str(Path(__file__).resolve().parents[1])]
    paths.extend(str(Path(part or '.').resolve())
                 for part in env.get('PYTHONPATH', '').split(os.pathsep))
    env['PYTHONPATH'] = os.pathsep.join(paths)
    result = subprocess.run([
        sys.executable, '-m', 'pytest', str(tmp_path), '-q',
        '-p', 'brainscore.validation.required_checks',
        '--umi-check-report', str(report),
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == code, result.stdout + result.stderr
    evidence = json.loads(report.read_text())
    assert len(evidence['passed']) == passed
    assert len(evidence['unexpected_skips']) == unexpected


def test_optional_permission_is_scoped_to_test_and_reason():
    from types import SimpleNamespace
    from brainscore.validation.required_checks import RequiredChecks
    plugin = RequiredChecks('unused.json', 'unified')
    node = 'tests/test_openpi_instrumentation.py'
    for nodeid, reason in [
        (node, 'Set OPENPI_SOURCE for optional OpenPI/JAX qualification'),
        (node, 'unexpected dependency failure'),
        ('tests/test_new_required.py', 'Set OPENPI_SOURCE for optional OpenPI/JAX qualification'),
    ]:
        plugin._record_skip(SimpleNamespace(nodeid=nodeid, longrepr=reason))
    assert [entry['optional'] for entry in plugin.skipped] == [True, False, False]


@pytest.mark.parametrize('failure', ['missing_report', 'timeout'])
def test_workspace_rejects_incomplete_run(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    from brainscore.validation import workspace
    root = tmp_path / 'repos'
    for repo in workspace.SUITES:
        (root / repo / 'tests').mkdir(parents=True)

    def git(command, **kwargs):
        if command[1] == 'rev-parse':
            return 'a' * 40
        return '' if kwargs.get('text') else b''

    def run(command, **kwargs):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(workspace.subprocess, 'check_output', git)
    monkeypatch.setattr(workspace.subprocess, 'run', run)
    out = tmp_path / 'report.json'
    assert workspace.qualify(root, out) == 1
    report = json.loads(out.read_text())
    assert all(row['returncode'] != 0 for row in report['repositories'].values())
