"""The recording plan of a session.

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
from brainscore_core.streaming import Session
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
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


class _Layered:
    """An extractor that lists its layers and logs each call."""

    def __init__(self, layers, calls):
        self.layers, self.calls = list(layers), calls

    def layer_names(self):
        return list(self.layers)

    def __call__(self, stimuli, layers, **kwargs):
        import numpy as np
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
        self.calls.append(('extract', tuple(layers), tuple(stimuli['stimulus_id'])))
        ids = list(stimuli['stimulus_id'])
        return NeuroidAssembly(np.ones((len(ids), len(layers))), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'stimulus_path': ('presentation', ids),
            'neuroid_id': ('neuroid', list(layers)), 'layer': ('neuroid', list(layers))})


def _pictures():
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    table = StimulusSet({'stimulus_id': ['p0', 'p1', 'p2'], 'image_path': ['p0.png', 'p1.png', 'p2.png']})
    table.identifier = 'plan-pictures'
    return table


@pytest.mark.parametrize('streaming', [False, True])
def test_the_region_order_and_the_shared_calls_are_decided_once_per_session(streaming, monkeypatch):
    """Before the plan: once for each recorded batch (1 for a table, n for a stream)."""
    from brainscore._model import channels
    ordered, original = [], channels._plan
    monkeypatch.setattr(channels, '_plan', lambda *a, **k: ordered.append(1) or original(*a, **k))
    calls = []
    model = BrainScoreModel('pictures', extractor={'vision': _Layered(['late', 'early'], calls)},
                            region_layer_map={'IT': 'late', 'V4': 'early'})
    session = stimulus_session(_pictures(), ['neural:IT', 'neural:V4'], streaming=streaming)
    model.participate(session)
    assert session.collect('neural:V4').sizes['presentation'] == 3
    assert len(ordered) == 1


def test_a_region_with_no_extractor_for_its_layers_is_refused_before_the_first_region_runs():
    """Before the plan: IT was recorded, then V4 was refused."""
    calls = []
    model = BrainScoreModel('two-forms', extractor={'vision': _Layered(['late'], calls),
                                                    'video': _Layered(['early'], calls)},
                            region_layer_map={'IT': 'late', 'V4': 'early'})
    with pytest.raises(ExecutionNotFeasibleError, match=r"has no 'vision' extractor .* layers \['early'\]"):
        model.participate(stimulus_session(_pictures(), ['neural:IT', 'neural:V4']))
    assert calls == []


# --- a source given with a table by its caller: each batch decides, as before the plan ---

class _Recordable:
    """An extractor that lists layers 'a' and 'b', can share a call, and records during a task."""
    independent_layers = True

    def __init__(self, name, log):
        self.name, self.log = name, log

    def layer_names(self):
        return ['a', 'b']

    def _assembly(self, ids, layers):
        import numpy as np
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
        return NeuroidAssembly(np.ones((len(ids), len(layers))), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', list(ids)), 'stimulus_path': ('presentation', list(ids)),
            'neuroid_id': ('neuroid', list(layers)), 'layer': ('neuroid', list(layers))})

    def __call__(self, stimuli, layers, **_):
        self.log.append((self.name, 'call', tuple(layers)))
        return self._assembly(list(stimuli['stimulus_id']), layers)

    def record(self, layers, stimuli):
        self.log.append((self.name, 'record', tuple(layers)))
        extractor = self

        class _Live:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def take(self, row):
                return {layer: extractor._assembly([row['stimulus_id']], [layer]) for layer in layers}
        return _Live()


def _given(table, *events):
    """A session whose source the caller gives together with a table (not ``stimulus_session``)."""
    def source():
        yield from events
    return lambda outputs, task=None: Session(source(), outputs, task, stimuli=table)


def _image_event(**meta):
    from brainscore_core.streaming import StreamEvent
    return StreamEvent('vision:image', 'p0.png', 0.0, {'stimulus_id': 's0', 'column': 'image_path', **meta})


def test_a_bad_row_that_a_given_source_never_presents_does_not_refuse_the_session():
    """The source presents row 0 only; row 1 has no text. Before the repair the plan checked row 1."""
    from brainscore_core.streaming import StreamEvent
    table = StimulusSet({'stimulus_id': ['s0', 's1'], 'sentence': ['hi', None]})
    table.identifier = 'given-bad-row'
    calls = []
    model = BrainScoreModel('replies', in_channels={'text'}, reply=lambda prompt: calls.append(prompt.stimulus_id) or 'cat')
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), present=('text',))
    session = _given(table, StreamEvent('text', 'hi', 0.0, {'stimulus_id': 's0', 'stimulus_index': 0,
                                                             'column': 'sentence'}))(['behavior:choice'], task)
    model.participate(session)
    assert calls == ['s0']


def _image_and_video_table():
    table = StimulusSet({'stimulus_id': ['s0'], 'image_path': ['p0.png'], 'video_path': ['c0.mp4']})
    table.identifier = 'given-image-and-video'
    return table


def test_an_image_source_with_an_image_and_video_table_records_with_the_image_extractor():
    """Before the repair the plan grouped the recording by the table, and opened the video recorder."""
    log = []
    model = BrainScoreModel('two-forms', extractor={'vision': _Recordable('image', log), 'video': _Recordable('video', log)},
                            region_layer_map={'IT': 'a'}, reply=lambda prompt: 'cat')
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    model.participate(_given(_image_and_video_table(), _image_event())(['behavior:choice', 'neural:IT'], task))
    assert log == [('image', 'record', ('a',))]


def test_an_image_source_with_an_image_and_video_table_shares_one_image_call():
    """Before the repair the plan shared the call on the video extractor, so the image extractor ran twice."""
    log = []
    model = BrainScoreModel('two-forms', extractor={'vision': _Recordable('image', log), 'video': _Recordable('video', log)},
                            region_layer_map={'IT': 'a', 'V4': 'b'})
    model.participate(_given(_image_and_video_table(), _image_event())(['neural:IT', 'neural:V4']))
    assert log == [('image', 'call', ('a', 'b'))]


def test_a_listed_image_source_with_an_image_and_text_table_records_with_one_call():
    """Before the repair the plan read two inputs from the table and refused the absent text extractor."""
    log = []
    table = StimulusSet({'stimulus_id': ['s0'], 'image_path': ['p0.png'], 'sentence': ['hi']})
    table.identifier = 'given-image-and-text'
    model = BrainScoreModel('pictures', extractor={'vision': _Recordable('image', log)}, in_channels={'text'},
                            region_layer_map={'IT': 'a'})
    session = Session([_image_event(stimulus_index=0)], ['neural:IT'], stimuli=table)
    model.participate(session)
    assert log == [('image', 'call', ('a',))]
    assert session.collect('neural:IT').sizes['neuroid'] == 1
