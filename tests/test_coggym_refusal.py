"""A CogGym trial the subject cannot serve ends the run with the subject's own reason.

Pre-flight cannot know what a single trial shows (for example how many pictures), so a
subject refuses such a trial at its call with ``ExecutionNotFeasibleError``. The run stops
there: no later trial reaches the subject, and the error names the trial and the reason.
"""
import pytest

import brainscore
from brainscore_core.compatibility import ExecutionNotFeasibleError

from .test_coggym import make_subject, reference, reply_with_trial  # noqa: F401  (reference is a fixture)

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

    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', checkout=root, reset=resets.append,
                                          repetitions=2, output_dir=tmp_path / 'run')
    with pytest.raises(ExecutionNotFeasibleError) as refused:
        brainscore.score(make_subject(reply), benchmark, check_mem=False)
    assert REASON in str(refused.value)
    assert len(calls) == 2           # no trial after the refused one reaches the subject
    assert resets == [1]             # and no later repetition starts
