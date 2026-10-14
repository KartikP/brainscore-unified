"""Exercise orchestration and safeguards without running a vision benchmark."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from brainscore.validation import run_vision_parity as launcher

pytestmark = pytest.mark.unit


def arguments(tmp_path):
    return ['--resnet18', 'unused.pt', '--gpt2', 'unused-gpt2', '--out', str(tmp_path / 'run')]


def test_plan_is_default_and_starts_no_process(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launcher, '_run_case', lambda *a, **kw: pytest.fail('Unexpected execution'))
    assert launcher.main(arguments(tmp_path)) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan['commands']) == 4
    assert not plan['vision_complete'] and not plan['release_complete']
    assert not (tmp_path / 'run').exists()
    for case, command in plan['commands'].items():
        assert command[command.index('--only') + 1] == case
        assert command[command.index('--device') + 1] == 'cpu'


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_invalid_limit_is_rejected(tmp_path, value):
    with pytest.raises(SystemExit):
        launcher.main([*arguments(tmp_path), '--timeout-per-case-s', value])


def test_memory_preflight_stops_before_first_process(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.psutil, 'virtual_memory', lambda: SimpleNamespace(available=1))
    monkeypatch.setattr(launcher, '_run_case', lambda *a, **kw: pytest.fail('Unexpected execution'))
    assert launcher.main([*arguments(tmp_path), '--execute']) == 1
    report = json.loads((tmp_path / 'run/summary.json').read_text())
    assert not report['vision_complete']
    assert next(iter(report['cases'].values()))['stop_reason'] == 'insufficient available RAM'


@pytest.mark.parametrize('failure', [None, 'missing', 'incomplete', 'failed', 'checkpoint'])
def test_sequential_reports_fail_closed(tmp_path, monkeypatch, failure):
    calls = []
    monkeypatch.setattr(launcher.psutil, 'virtual_memory',
                        lambda: SimpleNamespace(available=64 * launcher.GIB))

    def run_case(command, **kwargs):
        case = command[command.index('--only') + 1]
        calls.append(case)
        report = {
            'selection_complete': failure != 'incomplete',
            'release_complete': False, 'reference_only': False,
            'environment': {'requested_device': 'cpu'},
            'policy': launcher.CPU_FP32_POLICY, 'failures': {},
            'reports': {case: {'routes': {r: {} for r in ('legacy', 'adapter', 'native')}}},
            'checkpoints': {'resnet18': 'fixed' if failure != 'checkpoint' else len(calls)},
        }
        if failure != 'missing':
            Path(command[-1]).write_text(json.dumps(report))
        return {'returncode': 1 if failure == 'failed' else 0, 'stop_reason': None}

    monkeypatch.setattr(launcher, '_run_case', run_case)
    assert launcher.main([*arguments(tmp_path), '--execute']) == (0 if failure is None else 1)
    summary = json.loads((tmp_path / 'run/summary.json').read_text())
    assert summary['vision_complete'] is (failure is None)
    assert summary['release_complete'] is False
    expected_count = 4 if failure is None else 2 if failure == 'checkpoint' else 1
    assert calls == list(launcher.VISION_CASES[:expected_count])


@pytest.mark.skipif(os.name != 'posix', reason='Launcher uses POSIX process groups')
def test_real_child_is_stopped_on_timeout(tmp_path, monkeypatch):
    # Keep OS process-enumeration permissions separate from the real kill test.
    process = SimpleNamespace(
        children=lambda **kwargs: [], is_running=lambda: True,
        memory_info=lambda: SimpleNamespace(rss=0),
    )
    monkeypatch.setattr(launcher.psutil, 'Process', lambda pid: process)
    status = launcher._run_case(
        [sys.executable, '-c', 'import time; time.sleep(30)'],
        directory=tmp_path, timeout_s=0.02,
        max_rss_bytes=100 * launcher.GIB, min_available_bytes=0,
    )
    assert status['stop_reason'] == 'time limit'
    assert status['returncode'] != 0


@pytest.mark.skipif(os.name != 'posix', reason='Launcher uses POSIX process groups')
def test_monitoring_failure_stops_real_child(tmp_path, monkeypatch):
    def denied(pid):
        raise PermissionError('process enumeration denied')
    monkeypatch.setattr(launcher.psutil, 'Process', denied)
    status = launcher._run_case(
        [sys.executable, '-c', 'import time; time.sleep(30)'],
        directory=tmp_path, timeout_s=10,
        max_rss_bytes=100 * launcher.GIB, min_available_bytes=0,
    )
    assert status['stop_reason'].startswith('resource monitoring failed:')
    assert status['returncode'] != 0
