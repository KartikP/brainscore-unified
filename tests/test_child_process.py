"""CogGym's evaluator runs in a child process; each model call comes back to the parent.

Supplemental tests of the process helper itself (its protocol, deadlines and errors). The
guarantees a user depends on are tested through the public loaders in ``test_coggym*.py``.

A real child process, started with this interpreter, runs the CogGym worker on a stand-in
evaluator package in a temporary folder. No install, no network, no clone.
"""
from pathlib import Path
import sys
import time

import pytest

from brainscore.harnesses import _child_process
from brainscore.harnesses._child_process import ChildFailed, ChildProcess

from .coggym_stand_in import EXPERIMENT, write_checkout

pytestmark = pytest.mark.unit

WORKER = Path(_child_process.__file__).with_name('_coggym_worker.py')


def _worker(checkout, log_directory, idle_timeout_s=20.0):
    return ChildProcess([sys.executable, WORKER, checkout], log_directory=log_directory,
                        idle_timeout_s=idle_timeout_s)


def test_each_model_call_comes_back_to_the_parent_and_its_reply_reaches_the_evaluator(tmp_path):
    # The evaluator keeps the failure's type and how many runs this process has made.
    checkout = write_checkout(tmp_path / 'coggym', cli='''
def _run_experiment(experiment, selected, provider, model, temperature, max_tokens):
    RUNS.append(model)
    results = []
    for trial in selected:
        try:
            raw = provider.complete_with_metadata(*build_messages(trial, experiment), model,
                                                  temperature, max_tokens)
            results.append({'trial_id': trial.id, 'response': raw['text'], 'run': len(RUNS)})
        except Exception as error:
            results.append({'trial_id': trial.id, 'error': str(error), 'error_type': type(error).__name__})
    return results
''')
    calls = []

    def answer(call):
        calls.append(call)
        if call['messages'][0]['content'] == 'trial-1':
            return {'error': ['ValueError', 'the subject could not answer']}
        return {'reply': {'text': f"answer to {call['messages'][0]['content']}",
                          'reasoning': None, 'token_usage': None}}

    with _worker(checkout, tmp_path / 'log') as child:
        prepared = child.ask({'do': 'prepare', 'experiment': EXPERIMENT, 'trial_ids': None})
        first = child.ask({'do': 'run', 'model': 'm', 'temperature': 1.0, 'max_tokens': 7}, answer)
        second = child.ask({'do': 'run', 'model': 'm', 'temperature': 1.0, 'max_tokens': 7}, answer)
    assert [trial['id'] for trial in prepared['trials']] == prepared['canonical']
    assert prepared['trials'][0]['messages'] == [{'role': 'user', 'content': 'trial-0'}]
    # The call carries CogGym's arguments; the reply's text and a failure's type and message
    # reach the evaluator's trial record.
    assert calls[0] == {'system': 'system', 'messages': [{'role': 'user', 'content': 'trial-0'}],
                        'model': 'm', 'temperature': 1.0, 'max_tokens': 7}
    by_trial = {result['trial_id']: result for result in first['results']}
    assert by_trial['trial-0']['response'] == 'answer to trial-0'
    assert (by_trial['trial-1']['error_type'], by_trial['trial-1']['error']) == (
        'ValueError', 'the subject could not answer')
    assert first['summary'] == {'n_trials': len(first['results'])}
    # One process serves both repetitions: the evaluator's module state goes on.
    assert {result['run'] for result in second['results'] if 'run' in result} == {2}
    assert child.process.poll() is not None


def test_a_child_that_fails_a_command_reports_its_error_and_stays_usable(tmp_path):
    checkout = write_checkout(tmp_path / 'coggym', analyze='''
def analyze(paths, eml, manifest, min_items=5):
    raise OverflowError('the reference scorer overflowed')
''')
    with _worker(checkout, tmp_path / 'log') as child:
        child.ask({'do': 'prepare', 'experiment': EXPERIMENT, 'trial_ids': None})
        with pytest.raises(ChildFailed) as failed:
            child.ask({'do': 'analyze', 'paths': []})
        assert failed.value.error_type == 'OverflowError'
        assert 'overflowed' in str(failed.value)
        assert child.ask({'do': 'prepare', 'experiment': EXPERIMENT, 'trial_ids': ['trial-2']})['trials'][0]['id'] == 'trial-2'


def test_a_worker_that_cannot_start_reports_its_error(tmp_path):
    with pytest.raises(ChildFailed) as failed:
        with _worker(tmp_path / 'no-checkout', tmp_path / 'log') as child:
            child.ask({'do': 'prepare', 'experiment': EXPERIMENT, 'trial_ids': None})
    assert failed.value.error_type == 'FileNotFoundError'
    assert child.process.poll() is not None


def test_a_child_that_exits_gives_a_clear_error_with_its_log(tmp_path):
    with pytest.raises(ChildFailed) as failed:
        with _stand_in(tmp_path, 'sys.stderr.write("a fixture failure\\n"); sys.exit(3)\n') as child:
            child.ask({'do': 'prepare'})
    assert 'the child exited' in str(failed.value) and 'a fixture failure' in str(failed.value)
    assert child.process.poll() is not None


def test_a_silent_child_is_killed_and_reaped(tmp_path):
    checkout = write_checkout(tmp_path / 'coggym', cli='''
def build_messages(trial, experiment):
    import time
    time.sleep(60)
''')
    started = time.monotonic()
    with pytest.raises(ChildFailed, match='said nothing'):
        with _worker(checkout, tmp_path / 'log', idle_timeout_s=1.0) as child:
            child.ask({'do': 'prepare', 'experiment': EXPERIMENT, 'trial_ids': None})
    assert time.monotonic() - started < 10
    assert child.process.poll() is not None


def _stand_in(tmp_path, body):
    script = tmp_path / 'child.py'
    script.write_text('import os, sys, time\n' + body)
    return ChildProcess([sys.executable, script], log_directory=tmp_path / 'log', idle_timeout_s=0.5)


def test_a_child_that_reads_nothing_is_killed_at_the_deadline(tmp_path):
    started = time.monotonic()
    with pytest.raises(ChildFailed, match='read nothing'):
        with _stand_in(tmp_path, 'time.sleep(60)\n') as child:
            child.ask({'do': 'prepare', 'payload': 'x' * (4 * 1024 * 1024)})
    assert time.monotonic() - started < 10 and child.process.poll() is not None


def test_a_child_that_writes_a_line_slowly_is_killed_at_the_deadline(tmp_path):
    started = time.monotonic()
    body = ('out = os.fdopen(int(os.environ["BRAINSCORE_CHILD_FDS"].split(",")[1]), "wb", 0)\n'
            'while True:\n    out.write(b"x")\n    time.sleep(0.05)\n')
    with pytest.raises(ChildFailed, match='nothing complete'):
        with _stand_in(tmp_path, body) as child:
            child.ask({'do': 'prepare'})
    assert time.monotonic() - started < 10 and child.process.poll() is not None
