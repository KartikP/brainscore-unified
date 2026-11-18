"""No process of CogGym's evaluator outlives the load or the run that started it.

The stand-in evaluator starts a sleeping descendant (as an evaluator's own helper process
would) and writes its process id to a file; after the load or the run, that process is gone,
whether the evaluator finished, failed or exited.
"""
import os
import time

import pytest

from brainscore.harnesses import coggym

from tests.test_harnesses.test_coggym import reference  # noqa: F401  (a fixture)

pytestmark = pytest.mark.unit

DESCENDANT = '''
def _descendant():
    import subprocess, sys
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    with open({path!r}, 'a') as handle:
        handle.write(str(child.pid) + '\\n')
'''


def _gone(pid_file, within_s=5.0):
    """Whether every process id in ``pid_file`` is gone (allowing the system time to reap it)."""
    pids = [int(line) for line in pid_file.read_text().split()]
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            return bool(pids)
        time.sleep(0.05)
    for pid in alive:           # leave nothing behind, also when the test fails
        os.kill(pid, 9)
    return False


class _Provider:
    def complete_with_metadata(self, system, messages, model, temperature, max_tokens):
        return {'text': 'answer', 'reasoning': None, 'token_usage': None}


def test_a_helper_process_of_the_preflight_is_gone_after_the_load(reference, tmp_path):
    root, stand_in = reference
    pids = tmp_path / 'pids'
    stand_in.cli(DESCENDANT.format(path=str(pids)) + '''
_build = build_messages


def build_messages(trial, experiment):
    _descendant()
    return _build(trial, experiment)
''')
    coggym.CogGymRunner(root, model='fixture')
    assert _gone(pids)


def test_a_helper_process_of_a_finished_run_is_gone(reference, tmp_path):
    root, stand_in = reference
    pids = tmp_path / 'pids'
    runner = coggym.CogGymRunner(root, model='fixture')
    stand_in.cli(DESCENDANT.format(path=str(pids)) + '''
_aggregate = aggregate_experiment


def aggregate_experiment(rows):
    _descendant()
    return _aggregate(rows)
''')
    runner.run(_Provider(), output_dir=tmp_path / 'run', reset=lambda repetition: None)
    assert _gone(pids)


def test_a_helper_process_of_an_evaluator_that_exits_is_gone(reference, tmp_path):
    root, stand_in = reference
    pids = tmp_path / 'pids'
    runner = coggym.CogGymRunner(root, model='fixture')
    stand_in.cli(DESCENDANT.format(path=str(pids)) + '''
def aggregate_experiment(rows):
    import os
    _descendant()
    os._exit(3)
''')
    with pytest.raises(RuntimeError):         # the evaluator's process exited
        runner.run(_Provider(), output_dir=tmp_path / 'run', reset=lambda repetition: None)
    assert _gone(pids)


def test_a_failure_to_open_the_pipes_leaves_no_descriptor_open(reference, monkeypatch):
    """The system refuses the second pipe (too many open files): nothing stays open."""
    import errno
    root, _ = reference
    real, calls = os.pipe, []

    def pipe():
        # Only the pipes of the evaluator's process: the second of them is refused.
        import sys
        caller = sys._getframe(1)
        if (caller.f_code.co_filename.endswith('_child_process.py')
                and any('_coggym_worker.py' in part for part in caller.f_locals['self'].command)):
            calls.append(None)
            if len(calls) == 2:
                raise OSError(errno.EMFILE, 'Too many open files')
        return real()
    before = len(os.listdir('/dev/fd'))
    monkeypatch.setattr(os, 'pipe', pipe)
    with pytest.raises(OSError):
        coggym.CogGymRunner(root, model='fixture')
    monkeypatch.setattr(os, 'pipe', real)
    assert len(os.listdir('/dev/fd')) == before


def test_a_process_group_is_never_signalled_after_its_process_is_reaped(reference, tmp_path, monkeypatch):
    """Once a process is reaped its id can belong to another process, so its group is never
    signalled again: through a load, a finished run and a run whose evaluator exits."""
    root, stand_in = reference
    events = []
    real_waitpid, real_killpg = os.waitpid, os.killpg

    def waitpid(pid, options):
        result = real_waitpid(pid, options)
        if result[0]:
            events.append(('reaped', result[0]))
        return result

    def killpg(pgid, number):
        events.append(('signalled', pgid))
        return real_killpg(pgid, number)
    monkeypatch.setattr(os, 'waitpid', waitpid)
    monkeypatch.setattr(os, 'killpg', killpg)
    runner = coggym.CogGymRunner(root, model='fixture')
    runner.run(_Provider(), output_dir=tmp_path / 'finished', reset=lambda repetition: None)
    stand_in.cli('def aggregate_experiment(rows):\n    import os\n    os._exit(3)\n')
    with pytest.raises(RuntimeError):
        runner.run(_Provider(), output_dir=tmp_path / 'exited', reset=lambda repetition: None)
    reaped = set()
    for kind, pid in events:
        if kind == 'reaped':
            reaped.add(pid)
        else:
            assert pid not in reaped, f'the group {pid} was signalled after its process was reaped'
    assert reaped                    # the processes were reaped
