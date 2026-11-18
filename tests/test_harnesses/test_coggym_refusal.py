"""A CogGym trial the subject cannot serve ends the run with the subject's own reason.

Pre-flight cannot know what a single trial shows (for example how many pictures), so a
subject refuses such a trial at its call with ``ExecutionNotFeasibleError``. The run stops
there: no later trial reaches the subject, and the error names the trial and the reason.
"""
import pytest

import brainscore
from brainscore_core.compatibility import ExecutionNotFeasibleError

from tests.test_harnesses.test_coggym import make_subject, reference, reply_with_trial  # noqa: F401  (reference is a fixture)
from tests.test_benchmarks.test_coggym_benchmark import scored_reference  # noqa: F401  (a fixture)

pytestmark = pytest.mark.unit

REASON = 'this subject is shown at most one picture'


def test_the_first_trial_the_subject_cannot_serve_ends_the_run_with_its_reason(reference, tmp_path, monkeypatch):
    root, _ = reference
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path / 'cache'))
    calls, resets = [], []

    def reply(prompt):
        calls.append(prompt)
        if len(calls) == 2:          # the second trial cannot be served
            raise ExecutionNotFeasibleError(REASON)
        return reply_with_trial(prompt)

    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root, reset=resets.append,
                                          repetitions=2, output_dir=tmp_path / 'run')
    with pytest.raises(ExecutionNotFeasibleError) as refused:
        brainscore.score(make_subject(reply), benchmark, check_mem=False)
    assert REASON in str(refused.value)
    assert len(calls) == 2           # no trial after the refused one reaches the subject
    assert resets == [1]             # and no later repetition starts


def _same_prompt_for_every_trial(stand_in):
    """Several CogGym trials can show one prompt: here all of them do."""
    stand_in.cli("""
def build_messages(trial, experiment):
    return 'system', [{'role': 'user', 'content': 'one shared prompt'}]
""")


def test_a_refused_trial_is_named_when_several_trials_share_its_prompt(reference, tmp_path, monkeypatch):
    root, cli = reference
    _same_prompt_for_every_trial(cli)
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path / 'cache'))
    calls = []

    def reply(prompt):
        calls.append(prompt)
        if len(calls) == 2:          # the second trial, trial-1, cannot be served
            raise ExecutionNotFeasibleError(REASON)
        return 'an answer'

    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root, reset=lambda repetition: None,
                                          repetitions=2, output_dir=tmp_path / 'run')
    with pytest.raises(ExecutionNotFeasibleError) as refused:
        brainscore.score(make_subject(reply), benchmark, check_mem=False)
    assert 'trial-1' in str(refused.value) and 'trial-0' not in str(refused.value)
    assert REASON in str(refused.value)


def test_each_call_carries_its_own_trial_when_several_trials_share_a_prompt(
        reference, scored_reference, tmp_path, monkeypatch):
    root, cli = reference
    _same_prompt_for_every_trial(cli)
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path / 'cache'))
    shown = []

    def reply(prompt):
        shown.append(prompt.stimulus_id)
        return 'an answer'

    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root, reset=lambda repetition: None,
                                          repetitions=2, output_dir=tmp_path / 'run')
    score = brainscore.score(make_subject(reply), benchmark, check_mem=False)
    assert score.attrs['n_valid'] == score.attrs['n_trials'] == len(shown)
    first, second = shown[:len(shown) // 2], shown[len(shown) // 2:]
    assert len(set(first)) == len(first) and second == first     # one id per trial, again per repetition


def test_the_evaluator_records_the_refused_trial_as_failed_with_the_reason(reference, tmp_path, monkeypatch):
    """CogGym's own trial record of the refused trial (and of the trials after it) is a failure
    with the subject's reason, never an answer."""
    import json
    root, stand_in = reference
    seen = tmp_path / 'evaluator-results.json'
    stand_in.cli(f'''
_run = _run_experiment


def _run_experiment(*args):
    import json
    results = _run(*args)
    with open({str(seen)!r}, 'w') as handle:
        json.dump(results, handle)
    return results
''')
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path / 'cache'))
    calls = []

    def reply(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise ExecutionNotFeasibleError(REASON)
        return reply_with_trial(prompt)

    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root, reset=lambda repetition: None,
                                          repetitions=1, output_dir=tmp_path / 'run')
    with pytest.raises(ExecutionNotFeasibleError):
        brainscore.score(make_subject(reply), benchmark, check_mem=False)
    results = json.loads(seen.read_text())
    assert 'response' in results[0] and 'error' not in results[0]
    assert all('response' not in result and REASON in result['error'] for result in results[1:])
