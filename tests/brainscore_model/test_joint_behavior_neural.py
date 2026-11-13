"""One presentation, two measurements: behavior and neural:<region> together.

A toy network counts its forward passes, so these tests check that every
stimulus is presented once, and that the recorded activity is the activity of
the pass that produced the answer.
"""
from contextlib import contextmanager

import numpy as np
import pytest

from brainscore.brainscore_model import BrainScoreModel
from dataclasses import replace

from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
from brainscore_core.compatibility import CompatibilityError
from brainscore_core.contract import TaskContext
from brainscore_core.streaming_helpers import apply_state_change
from brainscore.selection import LayerSelector
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

LAYERS = {'early': 3, 'late': 2}   # layer -> number of units


class ToyNet:
    """Activity depends on the stimulus and on what else is in the prompt."""

    def __init__(self):
        self.passes, self.hooks, self.silenced, self.log = 0, [], False, []

    def forward(self, stimulus_id, prompt=''):
        self.passes += 1
        self.log.append(self.silenced)
        base = 0.0 if self.silenced else float(sum(map(ord, stimulus_id)) + len(prompt))
        out = {layer: base + np.arange(n, dtype=float) * (i + 1)
               for i, (layer, n) in enumerate(LAYERS.items())}
        for hook in self.hooks:
            hook(out)
        return out


def _assembly(rows, stimulus_ids, layers):
    values = np.array([np.concatenate([row[layer] for layer in layers]) for row in rows])
    return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', list(stimulus_ids)),
        'neuroid_id': ('neuroid', [f'{l}.{u}' for l in layers for u in range(LAYERS[l])]),
        'layer': ('neuroid', [l for l in layers for _ in range(LAYERS[l])])})


class ToyExtractor:
    identifier = backbone_id = 'toy'

    def __init__(self, net):
        self.net = net

    def __call__(self, stimuli, layers, **kwargs):
        ids = list(stimuli['stimulus_id'])
        return _assembly([self.net.forward(i) for i in ids], ids, layers)

    @contextmanager
    def record(self, layers, stimuli=None):
        captured = []
        recording = type('Recording', (), {})()

        def take(_row):
            first = captured[0]
            captured.clear()
            return {layer: _assembly([first], ['?'], [layer]) for layer in layers}
        recording.take = take
        self.net.hooks.append(captured.append)
        try:
            yield recording
        finally:
            self.net.hooks.remove(captured.append)


class NoRecordExtractor(ToyExtractor):
    record = property(lambda self: (_ for _ in ()).throw(AttributeError('record')))


STIMULI = StimulusSet({'stimulus_id': ['a', 'b', 'c'], 'sentence': ['x', 'y', 'z'],
                       'label': ['cat', 'dog', 'cat']})
STIMULI.identifier = 'joint-toy'


def _model(net, extractor_type=ToyExtractor, **kwargs):
    def answer(stimulus_row, instruction, label_set):
        out = net.forward(stimulus_row['stimulus_id'], prompt=instruction)
        return label_set[int(out['late'][0]) % 2]

    def intervention(change):
        net.silenced = True

        def cleanup():
            net.silenced = False
        return PerturbationApplied(handle_id=f'h{len(net.log)}', target=change.target,
                                   perturbation=change.perturbation, applied_at=0), cleanup
    return BrainScoreModel(identifier='toy', extractor={'text': extractor_type(net)},
                           region_layer_map={'A': LayerSelector('early', modality='text'),
                                             'B': LayerSelector('late', modality='text')},
                           answer=answer, intervention=intervention, **kwargs)


def _run(model, task, channels):
    session = stimulus_session(STIMULI, channels, task=task)
    model.participate(session)
    return session


GENERATE = TaskContext(task_type='label', label_set=['cat', 'dog'],
                       instruction='Name the animal.')


def test_answers_and_recording_come_from_one_presentation():
    net = ToyNet()
    model = _model(net)
    session = _run(model, GENERATE, ['behavior', 'neural:A', 'neural:B'])
    assert net.passes == len(STIMULI)                       # each stimulus shown once
    behavior = session.collect('behavior')
    model.reset()
    alone = _run(_model(ToyNet()), GENERATE, ['behavior']).collect('behavior')
    np.testing.assert_array_equal(behavior.values, alone.values)
    recorded = session.collect('neural:A')
    assert list(recorded['stimulus_id'].values) == ['a', 'b', 'c']
    assert set(recorded['region'].values) == {'A'}
    # The prompt is part of the pass: activity differs from passive viewing.
    expected = [ToyNet().forward(s, prompt='Name the animal.')['early'] for s in 'abc']
    np.testing.assert_array_equal(recorded.values, expected)


def test_readout_records_in_the_same_extraction():
    net = ToyNet()
    model = _model(net, behavioral_readout_layer='late')
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'],
                       fitting_stimuli=STIMULI)
    model.start_task(task)
    fitted = net.passes
    session = _run(model, task, ['behavior', 'neural:A'])
    assert net.passes - fitted == len(STIMULI)              # one extraction for both
    model.reset()
    reference = _model(ToyNet(), behavioral_readout_layer='late')
    alone = _run(reference, task, ['behavior']).collect('behavior')
    np.testing.assert_allclose(session.collect('behavior').values, alone.values)
    recorded = session.collect('neural:A')
    np.testing.assert_array_equal(recorded.values,
                                  [ToyNet().forward(s)['early'] for s in 'abc'])


def test_extractor_that_cannot_record_is_not_applicable_before_any_answer():
    net = ToyNet()
    model = _model(net, extractor_type=NoRecordExtractor)
    with pytest.raises(NotImplementedError, match='No session handler'):
        _run(model, GENERATE, ['behavior', 'neural:A'])
    assert net.passes == 0


def _form_stimuli(column):
    stimuli = StimulusSet({'stimulus_id': ['a', 'b', 'c'], column: ['a.x', 'b.x', 'c.x']})
    stimuli.identifier = f'joint-toy-{column}'
    return stimuli


@pytest.mark.parametrize('image_records', [True, False])
@pytest.mark.parametrize('form, column', [('vision', 'image_path'), ('video', 'video_path')])
def test_the_extractor_for_the_form_decides_recording(image_records, form, column):
    """The region is declared for both forms (the run serves each); during a task
    the run records with the extractor for the form of the stimuli, and an
    extractor that cannot record makes the pair N/A before any compute."""
    net = ToyNet()
    image, video = ((ToyExtractor, NoRecordExtractor) if image_records
                    else (NoRecordExtractor, ToyExtractor))

    def answer(stimulus_row, instruction, label_set):
        return label_set[int(net.forward(stimulus_row['stimulus_id'])['late'][0]) % 2]
    model = BrainScoreModel(identifier='forms', extractor={'vision': image(net), 'video': video(net)},
                            region_layer_map={'A': LayerSelector('early', modality=form)},
                            answer=answer)
    records = image_records == (form == 'vision')
    session = stimulus_session(_form_stimuli(column), ['behavior', 'neural:A'], task=GENERATE)
    model.check_session_support({'behavior', 'neural:A'})
    if records:
        model.participate(session)
        assert session.collect('neural:A').sizes['presentation'] == 3
    else:
        with pytest.raises(CompatibilityError, match='cannot record'):
            model.participate(session)
        assert net.passes == 0


def test_undeclared_region_is_not_supported():
    with pytest.raises(NotImplementedError):
        _run(_model(ToyNet()), GENERATE, ['behavior', 'neural:IT'])


def test_neural_session_after_a_task_returns_activity():
    model = _model(ToyNet())
    model.start_task(GENERATE)
    session = stimulus_session(STIMULI, ['neural:A'])
    model.participate(session)
    assert isinstance(session.collect('neural:A'), NeuroidAssembly)


SILENCE = StateChange(kind='ablation', target=Selection(layer='early'),
                      perturbation=Perturbation(kind='zero'))


def test_neural_only_session_under_a_task_records_during_the_task():
    net = ToyNet()
    session = _run(_model(net), GENERATE, ['neural:A'])
    assert net.passes == len(STIMULI)
    assert [e.channel for e in session.emitted] == ['neural:A']       # answers not sent
    recorded = session.collect('neural:A')
    expected = [ToyNet().forward(s, prompt='Name the animal.')['early'] for s in 'abc']
    np.testing.assert_array_equal(recorded.values, expected)
    assert recorded.attrs['recording']['task_type'] == 'label'
    assert recorded.attrs['recording']['forward_pass'] == 'prompt'


def test_passive_task_records_plain_viewing_unchanged():
    net = ToyNet()
    passive = TaskContext(task_type='passive')
    session = stimulus_session(STIMULI, ['neural:A'], task=passive)
    _model(net).participate(session)
    recorded = session.collect('neural:A')
    np.testing.assert_array_equal(recorded.values, [ToyNet().forward(s)['early'] for s in 'abc'])
    assert 'recording' not in recorded.attrs     # existing neural outputs are unchanged


def _readout_task(fit):
    return TaskContext(task_type='probabilities', label_set=['cat', 'dog'],
                       fitting_stimuli=STIMULI, readout_fit=fit)


def test_readout_fit_current_learns_on_the_changed_network():
    net = ToyNet()
    model = _model(net, behavioral_readout_layer='late')
    applied = apply_state_change(model, SILENCE)
    session = _run(model, _readout_task('current'), ['behavior', 'neural:A'])
    fitting, scoring = net.log[:len(STIMULI)], net.log[len(STIMULI):]
    assert fitting == [True] * len(STIMULI)
    assert scoring == [True] * len(STIMULI)                # tested on the changed network
    np.testing.assert_array_equal(session.collect('neural:A').values, [np.arange(3.)] * 3)  # silenced
    handle = applied.handle_id
    apply_state_change(model, StateChange(kind='reset', handle_id=handle))   # handle survives
    assert not net.silenced


def test_readout_fit_intact_refuses_a_change_it_cannot_switch_off():
    """The toy intervention has no switch; the fit stops before any pass."""
    net = ToyNet()
    model = _model(net, behavioral_readout_layer='late')
    apply_state_change(model, SILENCE)
    with pytest.raises(NotImplementedError, match='switched off in place'):
        _run(model, _readout_task('intact'), ['behavior', 'neural:A'])
    assert net.passes == 0 and net.silenced


def test_readout_fit_is_validated():
    with pytest.raises(ValueError, match='readout_fit'):
        _run(_model(ToyNet(), behavioral_readout_layer='late'), _readout_task('sometimes'),
             ['behavior'])
