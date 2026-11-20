"""Regions record on every task path: likelihood scoring, a reply with implied
probabilities, composite regions, streaming sessions, and extractors without
record() (through forward hooks on the model)."""
from contextlib import contextmanager

import numpy as np
import pytest

from brainscore.brainscore_model import BrainScoreModel
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore.selection import CompositeSelector, LayerSelector
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

STIMULI = StimulusSet({'stimulus_id': ['a', 'b', 'c'], 'sentence': ['x', 'yy', 'zzz']})
STIMULI.identifier = 'everywhere'


class Net:
    """Each pass sees a text; 'early' has 3 units, 'late' 2. Passes are logged."""

    def __init__(self):
        self.passes, self.hooks = [], []

    def forward(self, text):
        self.passes.append(text)
        out = {'early': np.array([len(text), 2.0 * len(text), 3.0 * len(text)]),
               'late': np.array([-len(text), 10.0])}
        for hook in self.hooks:
            hook(out)
        return out


class Recordable:
    identifier = 'toy'

    def __init__(self, net):
        self.net = net

    def __call__(self, stimuli, layers, **kwargs):
        raise AssertionError('recorded during the task')

    @contextmanager
    def record(self, layers, stimuli=None):
        seen = []

        class Take:
            @staticmethod
            def take(_row):
                first = seen[0]
                seen.clear()
                return {layer: NeuroidAssembly(first[layer][None], dims=['presentation', 'neuroid'], coords={
                    'stimulus_id': ('presentation', ['?']),
                    'neuroid_id': ('neuroid', [f'{layer}.{i}' for i in range(len(first[layer]))]),
                    'neuroid_num': ('neuroid', list(range(len(first[layer])))),
                    'layer': ('neuroid', [layer] * len(first[layer]))}) for layer in layers}
        self.net.hooks.append(seen.append)
        try:
            yield Take()
        finally:
            self.net.hooks.remove(seen.append)


def _model(net, region_map=None, replies=True):
    """A model with reply and likelihood parts; without ``replies`` it answers by likelihood."""
    def reply(prompt):
        net.forward(prompt.text())
        return 'cat'

    def likelihood(prompt, candidates):
        net.forward(prompt.text())                         # the prompt alone first
        return {c: -float(len(net.forward(prompt.text() + c)['late'])) - i
                for i, c in enumerate(candidates)}
    return BrainScoreModel('toy', extractor={'text': Recordable(net)},
                           region_layer_map=region_map or {'A': LayerSelector('early', modality='text')},
                           reply=reply if replies else None, likelihood=likelihood,
                           capability_config={'behavior-sessions': {
                               # The likelihood derives these formats; the statements declare them.
                               'behavior:probabilities': ('likelihood', 'softmax'),
                               **({} if replies else {'behavior:choice': ('likelihood', 'softmax_argmax')})}})


def _run(model, task, channels, streaming=False):
    session = stimulus_session(STIMULI, channels, task=task, streaming=streaming)
    model.participate(session)
    return session


def _prompt_text(stimulus_text):
    return f'Which?\n\n{stimulus_text}\n\nAnswer with exactly one of: cat, dog.'


def test_recording_during_likelihood_scoring_is_the_prompt_pass():
    net = Net()
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    recorded = _run(_model(net, replies=False), task, ['behavior:probabilities', 'neural:A']).collect('neural:A')
    expected = [Net().forward(_prompt_text(s))['early'] for s in ('x', 'yy', 'zzz')]
    np.testing.assert_array_equal(recorded.values, expected)


def test_log_probabilities_of_per_row_options_and_a_region_in_one_session():
    """Each row gives its own options (EWoK's form) and a region is recorded in the same pass:
    the recording keeps the scalar stimulus columns; the options stay with the answer."""
    stimuli = STIMULI.assign(options=[['cat', 'dog house'], ['dog', 'dog house'], ['cat', 'dog']])
    stimuli.identifier = 'everywhere-options'
    session = stimulus_session(stimuli, ['behavior:log_probabilities', 'neural:A'],
                               task=TaskContext(response=Choice('options', verbatim=True)))
    _model(Net()).participate(session)
    recorded = session.collect('neural:A')
    assert recorded.sizes['presentation'] == len(stimuli)
    assert list(recorded['sentence'].values) == list(stimuli['sentence'])
    log_probabilities = session.collect('behavior:log_probabilities')
    assert log_probabilities.attrs['response']['method'] == 'likelihood'
    assert set(log_probabilities['response_state'].values) == {'valid'}


class _PassiveToo(Recordable):
    """Records while answering, and presents a table passively through the shipped attachment."""

    def __call__(self, stimuli, layers, **kwargs):
        from brainscore.model_helpers.extractors import attach_stimulus_set
        values = [self.net.forward(text)[layers[0]] for text in stimuli['sentence']]
        return attach_stimulus_set(NeuroidAssembly(np.array(values), dims=['presentation', 'neuroid'], coords={
            'neuroid_id': ('neuroid', [f'{layers[0]}.{i}' for i in range(len(values[0]))]),
            'layer': ('neuroid', [layers[0]] * len(values[0]))}), stimuli)


def test_a_column_of_single_value_arrays_is_kept_on_a_recording_while_answering():
    """A zero-dimensional array is one value: the recording keeps it, as a passive recording does."""
    stimuli = STIMULI.assign(extra=[np.array(3), np.array(7), np.array(9)])
    stimuli.identifier = 'everywhere-single-values'
    net = Net()
    model = BrainScoreModel('toy', extractor={'text': _PassiveToo(net)},
                            region_layer_map={'A': LayerSelector('early', modality='text')},
                            likelihood=lambda prompt, labels: net.forward(prompt.text()) and {
                                label: -1.0 for label in labels})
    task = stimulus_session(stimuli, ['behavior:log_probabilities', 'neural:A'],
                            task=TaskContext(response=Choice(['cat', 'dog'], verbatim=True)))
    model.participate(task)
    passive = stimulus_session(stimuli, ['neural:A'])
    model.participate(passive)
    assert ([int(value) for value in task.collect('neural:A')['extra'].values]
            == [int(value) for value in passive.collect('neural:A')['extra'].values] == [3, 7, 9])


@pytest.mark.parametrize('streaming', [False, True])
def test_a_recording_while_answering_names_the_stimulus_columns_it_leaves_out(streaming):
    stimuli = STIMULI.assign(options=[['cat', 'dog house'], ['dog', 'dog house'], ['cat', 'dog']])
    stimuli.identifier = 'everywhere-left-out'
    session = stimulus_session(stimuli, ['behavior:log_probabilities', 'neural:A'],
                               task=TaskContext(response=Choice('options', verbatim=True)), streaming=streaming)
    _model(Net()).participate(session)
    recording = session.collect('neural:A').attrs['recording']
    assert 'options' in recording['stimulus_columns_left_out'] and recording['reason']
    # A table of single values leaves nothing out, and says nothing.
    scalar = stimulus_session(STIMULI, ['behavior:log_probabilities', 'neural:A'],
                              task=TaskContext(response=Choice(['cat', 'dog'], verbatim=True)), streaming=streaming)
    _model(Net()).participate(scalar)
    assert 'stimulus_columns_left_out' not in scalar.collect('neural:A').attrs['recording']


def test_a_reply_with_implied_probabilities_records_the_prompt_pass():
    net = Net()
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    session = _run(_model(net), task, ['behavior:choice', 'behavior:probabilities', 'neural:A'])
    expected = [Net().forward(_prompt_text(s))['early'] for s in ('x', 'yy', 'zzz')]
    np.testing.assert_array_equal(session.collect('neural:A').values, expected)
    assert session.collect('behavior:probabilities').attrs['response']['method'] == 'likelihood'


def test_a_composite_region_records_its_chosen_units():
    net = Net()
    region = CompositeSelector(layers=(('early', (0, 2)), ('late', None)), modality='text')
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    recorded = _run(_model(net, {'R': region}), task, ['behavior:choice', 'neural:R']).collect('neural:R')
    first = Net().forward(_prompt_text('x'))
    np.testing.assert_array_equal(recorded.values[0], [first['early'][0], first['early'][2], *first['late']])


def test_a_streaming_session_gets_each_trial_as_it_ends():
    net = Net()
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    session = _run(_model(net), task, ['behavior:choice', 'neural:A'], streaming=True)
    recorded = [e for e in session.emitted if e.channel == 'neural:A']
    assert len({e.meta['trial_index'] for e in recorded}) == len(STIMULI)   # one event per trial
    assert session.collect('neural:A').sizes['presentation'] == len(STIMULI)


@pytest.mark.parametrize('streaming', [True, False])
@pytest.mark.parametrize('timed', [True, False])
def test_an_answer_and_its_recording_carry_the_time_of_their_trial(streaming, timed):
    """Rule 4.2: time is the input's time, or None; never a trial counter.

    A streamed trial answers at the time of its input; the whole table, answered at
    once, is many trials and carries no time. The trial index is in meta.
    """
    stimuli = STIMULI.copy()
    if timed:
        stimuli['t_ms'] = [100.0, 600.0, 1100.0]
    stimuli.identifier = 'everywhere-timed' if timed else STIMULI.identifier
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    session = stimulus_session(stimuli, ['behavior:choice', 'neural:A'], task=task, streaming=streaming)
    _model(Net()).participate(session)
    assert session.emitted
    for event in session.emitted:
        if streaming:
            index = event.meta['trial_index']
            assert event.t_ms == (stimuli['t_ms'].iloc[index] if timed else None)
        else:
            assert event.t_ms is None


def test_an_extractor_without_record_is_recorded_through_the_model():
    torch = pytest.importorskip('torch')
    network = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))

    def extract(stimuli, layers, **kwargs):
        raise AssertionError('recorded during the task')

    def reply(prompt):
        network(torch.tensor([[float(len(prompt.text())), 1.0]]))
        return 'cat'
    model = BrainScoreModel('plain', model=network, extractor={'text': extract},
                            region_layer_map={'A': LayerSelector('0', modality='text')}, reply=reply)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    recorded = _run(model, task, ['behavior:choice', 'neural:A']).collect('neural:A')
    expected = network[0](torch.tensor([[float(len(_prompt_text('x'))), 1.0]])).detach().numpy()[0]
    np.testing.assert_allclose(recorded.values[0], expected, rtol=1e-6)
    assert not network[0]._forward_hooks


def test_a_streaming_session_answers_a_trial_before_it_reads_the_next():
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    session = stimulus_session(STIMULI, ['behavior:choice', 'neural:A'], task=task, streaming=True)
    log = []
    read, respond = session.next_input, session.respond

    def logged_read():
        event = read()
        if event is not None and event.channel != 'control':
            log.append('read')
        return event

    session.next_input = logged_read
    session.respond = lambda event: (log.append(event.channel), respond(event))[1]
    _model(Net()).participate(session)
    assert log == ['read', 'behavior:choice', 'neural:A'] * 3


@pytest.mark.parametrize('table_times', [None, [10.0, 20.0]])
def test_a_trial_answered_alone_keeps_the_time_of_its_input_events(table_times):
    """Rule 4.2: answered one trial at a time, a trial carries the time of the input
    events that presented it; a time column of an attached table never replaces it."""
    from brainscore_core.streaming import Session, StreamEvent
    input_times = [125.0, 875.0]
    events = [StreamEvent('text', text, t_ms, {'stimulus_id': sid, 'stimulus_index': index})
              for index, (sid, text, t_ms) in enumerate(zip('ab', ['x', 'yy'], input_times))]
    table = None
    if table_times is not None:
        table = StimulusSet({'stimulus_id': ['a', 'b'], 'sentence': ['x', 'yy'], 't_ms': table_times})
        table.identifier = 'listed-with-table'
    net = Net()

    def reply(prompt):
        net.forward(prompt.text())
        return 'cat'
    model = BrainScoreModel('toy', extractor={'text': Recordable(net)},
                            region_layer_map={'A': LayerSelector('early', modality='text')},
                            reply=reply, memory='trial_history')
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), trial_memory='history')
    session = Session(events, ['behavior:choice', 'neural:A'], task, stimuli=table)
    model.participate(session)
    assert {event.channel for event in session.emitted} == {'behavior:choice', 'neural:A'}
    for event in session.emitted:
        assert event.t_ms == input_times[event.meta['trial_index']]
