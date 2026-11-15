"""The recording plan of a session (step 10, chunks P3 and P4).

During a task or a policy step, the plan decides once which extractor records
which region (``plan_recordings``) and refuses a region that cannot be recorded
before any set-up or compute; before the plan, that check ran for every answered
batch and came after the readout fit. Without a task, the plan decides once the
order of the regions and the calls they share, and checks every region before
the first pass; before the plan, a region whose extractor was missing was
refused after the earlier regions were recorded.
"""
import pytest

from brainscore import BrainScoreModel
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from tests.brainscore_model import tiny_wrappers
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI, _features

pytestmark = pytest.mark.unit

REGION = 'V4'


def _tiny(folder, memory=None):
    wrapper, stimuli, regions, _ = tiny_wrappers.FACTORIES['text'](folder)
    layer = regions[REGION].name

    def reply(prompt):
        tiny_wrappers.show(wrapper, stimuli, prompt.stimulus_id, layer)
        return 'yes'
    model = BrainScoreModel('tiny-text', extractor={'text': wrapper}, region_layer_map={REGION: regions[REGION]},
                            reply=reply, memory=memory)
    return model, stimuli


@pytest.mark.parametrize('delivery', ['table', 'per trial', 'stream'])
def test_the_recording_groups_are_planned_once_per_session(delivery, tmp_path, monkeypatch):
    """Before the plan: once for each answered batch (1 for a table, n one at a time or streamed)."""
    from brainscore._model import recording
    planned, original = [], recording.plan_recordings
    monkeypatch.setattr(recording, 'plan_recordings', lambda *a, **k: planned.append(1) or original(*a, **k))
    model, stimuli = _tiny(tmp_path, memory='trial_history' if delivery == 'per trial' else None)
    task = TaskContext(instruction='Answer.', response=Choice(['yes', 'no']))
    session = stimulus_session(stimuli, ['behavior:choice', f'neural:{REGION}'], task,
                               streaming=delivery == 'stream')
    model.participate(session)
    assert session.collect(f'neural:{REGION}').sizes['presentation'] == len(stimuli)
    assert len(planned) == 1


def test_a_region_that_cannot_be_recorded_while_answering_is_refused_before_the_readout_fit():
    calls = []

    def answer(stimulus_row, instruction, label_set):
        calls.append('answer')
        return label_set[0]

    def extract(stimuli, layers, **kwargs):
        calls.append(('extract', tuple(layers)))
        return _features(stimuli, layers, **kwargs)
    model = BrainScoreModel('answers', extractor={'text': extract}, in_channels={'text'},
                            region_layer_map={'IT': 'late'}, behavioral_readout_layer='late', answer=answer,
                            capability_config={'behavior-sessions': {'behavior': 'generation'}})
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING)
    session = stimulus_session(STIMULI, ['behavior:choice', 'behavior:probabilities', 'neural:IT'], task)
    with pytest.raises(ExecutionNotFeasibleError, match=r"cannot record \['IT'\] while it answers"):
        model.participate(session)
    assert calls == []          # before the plan: the readout was fitted first
