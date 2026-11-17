"""A requested answer the task's response does not give is refused before any compute."""
import numpy as np
import pytest

from brainscore.brainscore_model import BrainScoreModel
from brainscore_core.compatibility import CompatibilityError, ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

STIMULI = StimulusSet({'stimulus_id': ['a', 'b', 'c', 'd'], 'sentence': ['w', 'x', 'y', 'z'],
                       'label': ['cat', 'dog', 'cat', 'dog']})
STIMULI.identifier = 'response-preflight-toy'


class _CountingExtractor:
    def __init__(self):
        self.calls = 0

    def __call__(self, stimuli, layers, **kwargs):
        self.calls += 1
        ids = list(stimuli['stimulus_id'])
        values = np.array([[1.0, 0.0] if i in 'ac' else [0.0, 1.0] for i in ids])
        return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['f0', 'f1']),
            'layer': ('neuroid', [layers[0]] * 2)})


def _readout_model():
    extractor = _CountingExtractor()
    model = BrainScoreModel(identifier='toy-readout', extractor={'text': extractor},
                            behavioral_readout_layer='late')
    return model, extractor


def _likelihood_model():
    calls = []

    def likelihood(prompt, candidates):
        calls.append(prompt.stimulus_id)
        return {c: float(i) for i, c in enumerate(candidates)}

    # The likelihood derives both formats; the statements declare them (rule 4.11).
    return BrainScoreModel(identifier='toy-likelihood', likelihood=likelihood, capability_config={
        'behavior-sessions': {'behavior:choice': ('likelihood', 'softmax_argmax'),
                              'behavior:probabilities': ('likelihood', 'softmax')}}), calls


def _ask(model, task, channels):
    session = stimulus_session(STIMULI, channels, task=task)
    model.participate(session)
    return session


def test_readout_refuses_a_format_the_response_does_not_give():
    model, extractor = _readout_model()
    fits = []
    model._behavioral.fit_behavioral_readout = lambda *a, **k: fits.append(1)  # never reached
    task = TaskContext(response=Choice(['cat', 'dog']),
                       fitting_stimuli=STIMULI)
    assert 'behavior:rating' in model.out_channels
    with pytest.raises(ValueError, match=r"behavior:rating.*behavior:choice"):
        _ask(model, task, ['behavior:rating'])
    assert extractor.calls == 0
    assert fits == []
    assert model._behavioral.readout_classifier is None


def test_likelihood_refuses_a_reaction_time_the_response_does_not_link():
    model, calls = _likelihood_model()
    task = TaskContext(instruction='Pick.', response=Choice(['cat', 'dog']))
    assert 'behavior:rt' in model.out_channels
    with pytest.raises(ValueError, match='behavior:rt'):
        _ask(model, task, ['behavior:choice', 'behavior:rt'])
    assert calls == []


def test_readout_matched_request_is_unchanged():
    model, extractor = _readout_model()
    task = TaskContext(response=Choice(['cat', 'dog']), fitting_stimuli=STIMULI)
    session = _ask(model, task, ['behavior:probabilities'])
    assert session.collect('behavior:probabilities').values.shape == (4, 2)
    assert extractor.calls > 0


def test_likelihood_matched_request_is_unchanged():
    model, calls = _likelihood_model()
    task = TaskContext(instruction='Pick.', response=Choice(['cat', 'dog']))
    session = _ask(model, task, ['behavior:choice', 'behavior:probabilities'])
    np.testing.assert_array_equal(session.collect('behavior:choice').values.argmax(1), [1, 1, 1, 1])
    assert session.collect('behavior:probabilities').values.shape == (4, 2)
    assert len(calls) == 4


def test_a_likelihood_alone_does_not_declare_bare_behavior():
    """Bare behavior answers a legacy task: only answer= or a readout with an extractor gives it."""
    model, calls = _likelihood_model()
    assert 'behavior' not in model.out_channels
    task = TaskContext(instruction='Pick.', response=Choice(['cat', 'dog']))
    with pytest.raises(ExecutionNotFeasibleError, match=r"does not produce \['behavior'\]"):
        _ask(model, task, ['behavior', 'behavior:choice', 'behavior:probabilities'])
    assert calls == []


def test_the_plan_refuses_addressed_channels_without_a_response_before_task_setup(monkeypatch):
    model, _ = _readout_model()
    setup = []
    monkeypatch.setattr(model, 'start_task', lambda *a, **k: setup.append('start_task'))
    session = stimulus_session(STIMULI, ['behavior:probabilities'], task=TaskContext(task_type='probabilities'))
    with pytest.raises(ExecutionNotFeasibleError, match='TaskContext'):
        model.participate(session)
    assert setup == []


def test_readout_without_a_response_refuses_addressed_channels_before_fitting():
    model, extractor = _readout_model()
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'], fitting_stimuli=STIMULI)
    with pytest.raises(ExecutionNotFeasibleError, match='TaskContext'):
        _ask(model, task, ['behavior:probabilities'])
    assert extractor.calls == 0
    assert model._behavioral.readout_classifier is None


class _RecordingExtractor(_CountingExtractor):
    """An extractor with record(layers, stimuli): it can record while a reply runs."""

    def record(self, layers, stimuli=None):
        raise AssertionError('pre-flight only; nothing runs')


@pytest.mark.parametrize('records, network_layers, supported', [
    (False, None, False),          # plain extractor, no network
    (True, None, True),            # the extractor records
    (False, ('late',), True),      # hooks on model= reach the region's layer
    (False, ('early',), False),    # model= lacks the region's layer
])
def test_recording_during_a_reply_is_decided_in_pre_flight(records, network_layers, supported):
    import torch
    network = None
    if network_layers is not None:
        network = torch.nn.Module()
        for name in network_layers:
            network.add_module(name, torch.nn.Linear(2, 2))
    extractor = _RecordingExtractor() if records else _CountingExtractor()
    model = BrainScoreModel(identifier='toy-reply', extractor={'text': extractor}, model=network,
                            region_layer_map={'IT': 'late'}, reply=lambda prompt: 'cat')
    channels = {'behavior:choice', 'neural:IT'}
    if supported:
        assert model.check_session_support(channels).identifier == 'behavior-sessions'
    else:
        with pytest.raises(ExecutionNotFeasibleError, match='no one part of it answers'):
            model.check_session_support(channels)
    assert extractor.calls == 0


def test_a_policy_records_a_region_only_where_the_wiring_can():
    import torch
    network = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model = BrainScoreModel(identifier='toy-policy', model=network, extractor={'vision': _CountingExtractor()},
                            region_layer_map={'V4': '0', 'IT': 'head'}, policy=lambda row: {'motor': 0})
    assert model.check_session_support({'motor', 'neural:V4'}).identifier == 'motor-sessions'
    with pytest.raises(ExecutionNotFeasibleError, match='no one part of it answers'):
        model.check_session_support({'motor', 'neural:IT'})


# ── A subject that cannot serve a session is N/A, by every path ──────────────

def _audio_model(policy=None):
    """A sound model with a region and no answering part, as ``random-wav2vec2-base``."""
    return BrainScoreModel(identifier='toy-audio', region_layer_map={'A1': 'layer'},
                           extractor={'audio': lambda stimuli, layers=None, **kwargs: None},
                           policy=policy)


@pytest.mark.parametrize('policy, outputs, missing', [
    (None, ('behavior:choice', 'neural:A1'), 'behavior:choice'),                  # no part at all
    (lambda row: {'motor': 0}, ('behavior:choice', 'neural:A1'), 'behavior:choice'),  # a part for another output
    (None, ('neural:V9',), 'neural:V9'),                                          # a region it does not map
], ids=['no-part', 'part-for-another-output', 'unmapped-region'])
def test_a_subject_that_cannot_serve_a_session_is_refused_before_compute(policy, outputs, missing):
    import re
    from types import SimpleNamespace
    import brainscore
    model = _audio_model(policy)
    named = re.escape(f"does not produce ['{missing}']")
    with pytest.raises(ExecutionNotFeasibleError, match=named):
        model.check_session_support(outputs)
    stimuli = StimulusSet({'stimulus_id': ['a'], 'audio_path': ['/no/such/file.wav']})
    stimuli.identifier = 'toy-audio-stimuli'
    task = (TaskContext(instruction='Is it loud?', response=Choice(['yes', 'no']))
            if 'behavior:choice' in outputs else None)
    with pytest.raises(ExecutionNotFeasibleError, match=named):
        model.participate(stimulus_session(stimuli, list(outputs), task))
    # Through score(), pre-flight refuses the same pair from the declarations.
    benchmark = SimpleNamespace(identifier='toy-audio-benchmark', version=1,
                                required_input_channels=frozenset({'audio'}),
                                requested_output_channels=frozenset(outputs))
    result = brainscore.score(model, benchmark, check_mem=False)
    assert result.attrs['status'] == 'N/A'
    assert re.search(re.escape(f"'{missing}'"), result.attrs['reason'])


def test_an_unknown_channel_is_refused_by_the_session_and_incompatible_in_pre_flight():
    from types import SimpleNamespace
    from brainscore_core.compatibility import check_compatibility
    model = _audio_model()
    with pytest.raises(ExecutionNotFeasibleError, match=r"does not produce \['telepathy'\]"):
        model.check_session_support({'telepathy'})
    benchmark = SimpleNamespace(identifier='toy', required_input_channels=frozenset({'audio'}),
                                requested_output_channels=frozenset({'telepathy'}))
    with pytest.raises(CompatibilityError, match="unknown channel 'telepathy'"):
        check_compatibility(model, benchmark)


def test_score_raises_when_no_part_answers_the_outputs_together():
    """Declarations pass; the subject refuses at session start; score() raises, no pass."""
    import brainscore
    extractor = _CountingExtractor()     # cannot record while a reply runs
    replies = []
    model = BrainScoreModel(identifier='toy-reply', extractor={'text': extractor},
                            region_layer_map={'IT': 'late'},
                            reply=lambda prompt: replies.append(prompt) or 'cat')

    class _Benchmark:
        identifier = 'toy-reply-and-record'
        required_input_channels = frozenset({'text'})
        requested_output_channels = frozenset({'behavior:choice', 'neural:IT'})

        def __call__(self, subject):
            task = TaskContext(instruction='Cat or dog?', response=Choice(['cat', 'dog']))
            subject.participate(stimulus_session(STIMULI, sorted(self.requested_output_channels), task))
            raise AssertionError('the session must not run')

    with pytest.raises(ExecutionNotFeasibleError, match="task type 'choice'.*no one part of it answers"
                       ".*passed the declared compatibility check") as caught:
        brainscore.score(model, _Benchmark(), check_mem=False)
    assert caught.value.channels == {'behavior:choice', 'neural:IT'}
    assert replies == [] and extractor.calls == 0
