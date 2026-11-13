"""A region is recorded during any task, through each extractor Brain-Score ships.

For each of the five wrappers, a tiny seeded network answers a task (a choice, a
text, a policy step) while the session requests one ``neural:<region>``. The
answering part shows each stimulus to the network the way the wrapper does,
so the recording must equal, exactly, the passive recording of the same
stimulus: values, neuroid coordinates and the extra axis (time or part).
The one stated difference: the time bins of a ``VideoWrapper`` task recording
have no coordinates (same number of bins, same values), because the times in
ms need the clip's duration, which the forward pass does not carry.
"""
import numpy as np
import pytest

from brainscore import BrainScoreModel
from brainscore_core.compatibility import CompatibilityError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Text
from brainscore_core.streaming import Session
from brainscore_core.streaming_helpers import EnvironmentSource, stimulus_session

from tests.brainscore_model import tiny_wrappers

pytestmark = pytest.mark.unit
pytest.importorskip('cv2')

REGION = 'V4'


def _build(kind, folder, **parts):
    wrapper, stimuli, regions, _ = tiny_wrappers.FACTORIES[kind](folder)
    layer = regions[REGION].name

    def reply(prompt):
        tiny_wrappers.show(wrapper, stimuli, prompt.stimulus_id, layer)
        return 'yes'

    def policy(row):
        tiny_wrappers.show(wrapper, stimuli, row['observation']['stimulus_id'], layer)
        return {'motor': 0}
    chosen = {'reply': reply, 'policy': policy}
    wired = {name: chosen[name] if value is True else value for name, value in parts.items()}
    model = BrainScoreModel(f'tiny-{kind}', extractor={tiny_wrappers.MODALITY[kind]: wrapper},
                            region_layer_map={REGION: regions[REGION]}, **wired)
    return model, stimuli


def _passive(kind, folder):
    model, stimuli = _build(kind, folder)
    session = stimulus_session(stimuli, [f'neural:{REGION}'])
    model.participate(session)
    return session.collect(f'neural:{REGION}')


def _names(assembly, dim):
    """Every coordinate on ``dim``: the levels of its index and its plain coordinates."""
    index = assembly.indexes.get(dim)
    names = [name for name in getattr(index, 'names', ()) if name and name != dim]
    return names + [name for name, coord in assembly.coords.items()
                    if coord.dims == (dim,) and name != dim and name not in names]


def assert_recorded_as_passive(recorded, passive, kind):
    assert recorded.dims == passive.dims
    np.testing.assert_array_equal(np.asarray(recorded.values), np.asarray(passive.values))
    assert _names(passive, 'neuroid')
    for name in _names(passive, 'neuroid'):
        np.testing.assert_array_equal(recorded[name].values, passive[name].values, err_msg=name)
    for dim in passive.dims:
        if dim in ('presentation', 'neuroid'):
            continue
        assert _names(passive, dim)
        if kind == 'video':
            assert not _names(recorded, dim)        # no clip duration during a task
            continue
        for name in _names(passive, dim):
            np.testing.assert_array_equal(recorded[name].values, passive[name].values, err_msg=name)


class _Rows:
    """An environment that shows each stimulus once, then ends."""

    def __init__(self, ids):
        self.ids, self.index = list(ids), 0

    def reset(self):
        self.index = 0
        return {'observation': {'stimulus_id': self.ids[0]}}

    def step(self, outputs):
        self.index += 1
        if self.index == len(self.ids):
            return {'observation': {'stimulus_id': self.ids[-1]}}, 'terminal'
        return {'observation': {'stimulus_id': self.ids[self.index]}}, None


ANSWERS = {
    'choice': (TaskContext(instruction='Is it there?', response=Choice(['yes', 'no'])), 'behavior:choice'),
    'text': (TaskContext(instruction='Describe it.', response=Text()), 'behavior:text'),
}


@pytest.mark.parametrize('answer', list(ANSWERS))
@pytest.mark.parametrize('kind', tiny_wrappers.KINDS)
def test_a_region_recorded_during_an_answer_equals_passive_viewing(tmp_path, kind, answer):
    task, channel = ANSWERS[answer]
    model, stimuli = _build(kind, tmp_path, reply=True)
    session = stimulus_session(stimuli, [channel, f'neural:{REGION}'], task=task)
    model.participate(session)
    recorded = session.collect(f'neural:{REGION}')
    assert list(recorded['stimulus_id'].values) == list(stimuli['stimulus_id'])
    assert_recorded_as_passive(recorded, _passive(kind, tmp_path), kind)


@pytest.mark.parametrize('kind', tiny_wrappers.KINDS)
def test_a_region_recorded_during_a_policy_step_equals_passive_viewing(tmp_path, kind):
    model, stimuli = _build(kind, tmp_path, policy=True)
    session = Session(EnvironmentSource(_Rows(stimuli['stimulus_id'])), ('motor', f'neural:{REGION}'))
    model.participate(session)
    recorded = session.collect(f'neural:{REGION}')
    assert recorded.sizes['presentation'] == len(stimuli)
    assert_recorded_as_passive(recorded, _passive(kind, tmp_path), kind)


# --- the two failures that were errors at run time --------------------------

def test_a_readout_and_an_answering_part_record_on_the_answering_path(tmp_path):
    """Before: the readout made pre-flight accept the region; the task took the
    reply, and AudioWrapper, with no record(layers), failed at run time."""
    model, stimuli = _build('audio', tmp_path, reply=True, behavioral_readout_layer='head')
    task = TaskContext(instruction='Is it there?', response=Choice(['yes', 'no']), prefer_path='generation')
    session = stimulus_session(stimuli, ['behavior:choice', f'neural:{REGION}'], task=task)
    model.participate(session)
    assert session.collect(f'neural:{REGION}').attrs['recording']['path'] == 'generation'
    assert_recorded_as_passive(session.collect(f'neural:{REGION}'), _passive('audio', tmp_path), 'audio')


def test_a_neural_only_session_under_a_task_records_during_the_task(tmp_path):
    """Before: the task path opened a recording on VideoWrapper, which had no record(layers)."""
    model, stimuli = _build('video', tmp_path, reply=True)
    task = ANSWERS['choice'][0]
    session = stimulus_session(stimuli, [f'neural:{REGION}'], task=task)
    model.participate(session)
    assert [event.channel for event in session.emitted] == [f'neural:{REGION}'] * 1
    recorded = session.collect(f'neural:{REGION}')
    assert recorded.attrs['recording']['forward_pass'] == 'prompt'
    assert_recorded_as_passive(recorded, _passive('video', tmp_path), 'video')


def test_a_subject_with_no_task_part_records_plain_viewing_under_a_task(tmp_path):
    """Nothing in the subject does a task, so the task does not change it."""
    model, stimuli = _build('audio', tmp_path)
    session = stimulus_session(stimuli, [f'neural:{REGION}'], task=ANSWERS['choice'][0])
    model.participate(session)
    recorded = session.collect(f'neural:{REGION}')
    passive = _passive('audio', tmp_path)
    assert recorded.identical(passive)


class _Plain:
    """A callable extractor with no record(layers) and no network passed as model=."""
    identifier = backbone_id = 'plain'

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, stimuli, layers, **kwargs):
        return self.inner(stimuli, layers=layers)


@pytest.mark.parametrize('channels', [['behavior:choice', f'neural:{REGION}'], [f'neural:{REGION}']],
                         ids=['joint', 'neural-only'])
def test_a_region_that_cannot_record_during_the_answer_is_not_applicable_before_compute(tmp_path, channels):
    wrapper, stimuli, regions, network = tiny_wrappers.audio(tmp_path)
    passes = []
    network.register_forward_pre_hook(lambda module, args: passes.append(1))
    model = BrainScoreModel('plain-audio', extractor={'audio': _Plain(wrapper)},
                            region_layer_map={REGION: regions[REGION]}, behavioral_readout_layer='head',
                            reply=lambda prompt: 'yes')
    task = TaskContext(instruction='Is it there?', response=Choice(['yes', 'no']), prefer_path='generation')
    with pytest.raises(CompatibilityError, match='cannot record'):
        model.participate(stimulus_session(stimuli, channels, task=task))
    assert passes == []
