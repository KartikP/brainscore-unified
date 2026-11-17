"""No process of CogGym's evaluator outlives the load or the run that started it.

The stand-in evaluator starts a sleeping descendant (as an evaluator's own helper process
would) and writes its process id to a file; after the load or the run, that process is gone,
whether the evaluator finished, failed or exited.
"""
import os
import time

import pytest

from brainscore.harnesses import coggym
from brainscore.harnesses._child_process import ChildFailed

from .test_coggym import reference  # noqa: F401  (a fixture)

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
    with pytest.raises(ChildFailed):
        runner.run(_Provider(), output_dir=tmp_path / 'run', reset=lambda repetition: None)
    assert _gone(pids)
