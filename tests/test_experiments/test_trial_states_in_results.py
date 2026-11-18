"""ExperimentResult.trials() shows the state core stored for each trial and its reason,
from the finalized answers the runner saves after each session."""
import pytest

from brainscore import BrainScoreModel
from brainscore_core import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.experiments import (Experiment, ExperimentResult, RecordBehavior,
                                    RecordInputsOutputs, TaskProtocol)
from brainscore.run_record import RunRecord

pytestmark = pytest.mark.unit

STIMULI = StimulusSet({'stimulus_id': ['cat', 'boat', 'unknown'],
                       'sentence': ['cat', 'boat', 'ambiguous']})
STIMULI.identifier = 'trial-states'


def _run(response, replies, directory, conditions=('baseline',), tools=None):
    def reply(prompt, _seen=[]):
        _seen.append(prompt)
        return replies[(len(_seen) - 1) % len(replies)]
    subject = BrainScoreModel(identifier='scripted', in_channels={'text'}, reply=reply)
    task = TaskContext(instruction='Respond to the word.', response=response)
    return Experiment(subject=subject,
                      protocol=TaskProtocol(STIMULI, task, conditions=list(conditions), repetitions=[0]),
                      tools=tools or [RecordBehavior(), RecordInputsOutputs()],
                      output_dir=directory).run()


def test_valid_invalid_and_missing_trials_show_their_states_and_reasons(tmp_path):
    result = _run(Choice(['animal', 'vehicle']), ['animal', 'nonsense', None], tmp_path / 'run')
    trials = result.trials()
    assert list(trials['choice_state']) == ['valid', 'invalid', 'missing']
    assert list(trials['choice_reason']) == ['', 'no label in the reply', 'no answer']
    assert len(trials) == 3                                   # no row from the finalized answers
    # A saved run read back gives the same table.
    reopened = ExperimentResult.open(tmp_path / 'run').trials()
    assert reopened.equals(trials)


@pytest.mark.parametrize('response, replies, column, states', [
    (Scale(1, 7, 1), ['6', '9', None], 'rating', ['valid', 'invalid', 'missing']),
    (Text(), ['A cat.', 'A boat.', None], 'text', ['valid', 'valid', 'missing'])])
def test_every_behavior_format_has_the_two_columns(tmp_path, response, replies, column, states):
    trials = _run(response, replies, tmp_path / 'run').trials()
    assert list(trials[f'{column}_state']) == states


def test_each_condition_reads_its_own_states(tmp_path):
    trials = _run(Choice(['animal', 'vehicle']), ['animal', 'vehicle', 'nonsense'], tmp_path / 'run',
                  conditions=('baseline', 'again')).trials()
    assert list(trials['condition']) == ['baseline'] * 3 + ['again'] * 3
    assert list(trials['choice_state']) == ['valid', 'valid', 'invalid'] * 2


def test_the_input_output_record_holds_no_finalized_answer(tmp_path):
    """Replay rebuilds the exchange from the input/output record; the finalized answers
    are kept only in the behavior record."""
    result = _run(Choice(['animal', 'vehicle']), ['animal', 'nonsense', None], tmp_path / 'run')
    kinds = {row['payload']['kind'] for row in RunRecord(tmp_path / 'run' / 'inputs_outputs').events()}
    assert 'answer' not in kinds
    behavior = [row['payload']['kind'] for row in RunRecord(tmp_path / 'run' / 'behavior').events()]
    assert behavior.count('answer') == 1


def test_a_session_that_ends_with_an_error_saves_no_finalized_answer(tmp_path):
    def reply(prompt):
        raise RuntimeError('the model failed')
    subject = BrainScoreModel(identifier='failing', in_channels={'text'}, reply=reply)
    task = TaskContext(instruction='Respond to the word.', response=Choice(['animal', 'vehicle']))
    with pytest.raises(RuntimeError, match='the model failed'):
        Experiment(subject=subject, protocol=TaskProtocol(STIMULI, task, repetitions=[0]),
                   tools=[RecordBehavior()], output_dir=tmp_path / 'run').run()
    kinds = [row['payload']['kind'] for row in RunRecord(tmp_path / 'run' / 'behavior').events()]
    assert 'answer' not in kinds


def test_a_session_whose_history_limit_dropped_responses_publishes_none():
    """collect() refuses such a session; the runner then saves no finalized answer."""
    from types import SimpleNamespace
    from brainscore.experiments.runner import _publish_finalized_answers
    published = []
    session = SimpleNamespace(dropped=2, outputs=('behavior:choice',), emitted=[],
                              collect=lambda *a: pytest.fail('collect must not be called'))
    _publish_finalized_answers(session, SimpleNamespace(publish=lambda *a, **k: published.append(a)))
    assert published == []


# --- one row per presented trial and occurrence, from the finalized answers ---------

import numpy as np

from brainscore_core import Subject
from brainscore_core.response_formats import response_record
from brainscore_core.streaming import StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

CHOICE = Choice(['animal', 'vehicle'])


class _Cells(Subject):
    """Answers each trial with one cell event (``answers[stimulus_id]``; None: no answer)."""
    identifier = 'cells'
    in_channels = {'text'}
    required_channels = set()

    def __init__(self, channel, answers, units=None):
        self.channel, self.answers, self.units = channel, answers, units

    @property
    def out_channels(self):
        # A choice task needs the subject to declare the choice; it answers what is requested.
        return {self.channel, 'behavior:choice'}

    def participate(self, session):
        seen = {}
        while (event := session.next_input()) is not None:
            sid = event.meta.get('stimulus_id')
            if sid is None:
                continue
            k = seen.get(sid, 0)
            seen[sid] = k + 1
            answer = self.answers.get((sid, k), self.answers.get(sid))
            if answer is None:
                continue
            meta = {'stimulus_id': sid, **({'units': self.units} if self.units else {})}
            session.respond(StreamEvent(self.channel, answer, event.t_ms, meta))

    def reset(self):
        pass


def _experiment(subject, task, directory, stimuli=STIMULI, conditions=('baseline',), repetitions=(0,),
                formats=None):
    return Experiment(subject=subject,
                      protocol=TaskProtocol(stimuli, task, conditions=list(conditions),
                                            repetitions=list(repetitions)),
                      tools=[RecordBehavior(formats)], output_dir=directory).run()


def _rows(trials, *columns):
    """The table's rows; an empty cell (None or NaN, as pandas keeps it) reads as None."""
    def cell(value):
        return None if value is None or (isinstance(value, float) and np.isnan(value)) else value
    return [tuple(cell(v) for v in row) for row in trials[list(columns)].itertuples(index=False)]


def test_a_batch_answer_with_one_omitted_trial_gives_three_rows(tmp_path):
    class _Batch(Subject):
        identifier, in_channels, out_channels, required_channels = 'batch', {'text'}, {'behavior:choice'}, set()

        def participate(self, session):
            while session.next_input() is not None:
                pass
            session.respond(StreamEvent('behavior:choice', BehavioralAssembly(
                np.array([[1, 0], [np.nan, np.nan]], dtype=np.float32), dims=['presentation', 'choice'],
                coords={'stimulus_id': ('presentation', ['cat', 'boat']), 'choice': ['animal', 'vehicle'],
                        'raw_answer': ('presentation', ['animal', 'a ship']),
                        'answer_valid': ('presentation', [True, False]),
                        'parse_error': ('presentation', ['', 'no label in the reply'])},
                attrs={'response': response_record('behavior:choice', 'test')}), None))

        def reset(self):
            pass
    trials = _experiment(_Batch(), TaskContext(response=CHOICE), tmp_path / 'run').trials()
    assert _rows(trials, 'stimulus_id', 'sentence', 'choice', 'choice_state', 'choice_reason',
                 'choice_raw', 'rows_from') == [
        ('cat', 'cat', 'animal', 'valid', '', 'animal', 'finalized answers'),
        ('boat', 'boat', None, 'invalid', 'no label in the reply', 'a ship', 'finalized answers'),
        ('unknown', 'ambiguous', None, 'missing', 'no answer event', None, 'finalized answers')]


def test_three_one_row_answer_events_give_three_rows(tmp_path):
    subject = _Cells('behavior:choice', {'cat': 'animal', 'boat': 'vehicle', 'unknown': 'animal'})
    trials = _experiment(subject, TaskContext(response=CHOICE), tmp_path / 'run').trials()
    assert _rows(trials, 'stimulus_id', 'choice', 'choice_state') == [
        ('cat', 'animal', 'valid'), ('boat', 'vehicle', 'valid'), ('unknown', 'animal', 'valid')]


def test_the_trial_history_case_of_the_review(tmp_path):
    """A subject with trial history answers one trial at a time: 'x', 'nonsense', None."""
    replies = iter(['x', 'nonsense', None])
    model = BrainScoreModel('remembers', in_channels={'text'}, reply=lambda prompt: next(replies),
                            memory='trial_history')
    stimuli = StimulusSet({'stimulus_id': ['a', 'b', 'c'], 'sentence': ['one', 'two', 'three']})
    stimuli.identifier = 'trial-states-history'
    task = TaskContext(instruction='Answer.', response=Choice(['x', 'y']), trial_memory='history')
    trials = _experiment(model, task, tmp_path / 'run', stimuli=stimuli).trials()
    assert _rows(trials, 'stimulus_id', 'choice', 'choice_state', 'choice_reason', 'choice_raw') == [
        ('a', 'x', 'valid', '', 'x'),
        ('b', None, 'invalid', 'no label in the reply', 'nonsense'),
        ('c', None, 'missing', 'no answer', None)]


def test_two_conditions_and_two_repetitions_give_twelve_rows(tmp_path):
    subject = _Cells('behavior:choice', {'cat': 'animal', 'boat': 'vehicle'})
    trials = _experiment(subject, TaskContext(response=CHOICE), tmp_path / 'run',
                         conditions=('baseline', 'again'), repetitions=(0, 1)).trials()
    assert len(trials) == 12
    assert _rows(trials, 'condition', 'repetition', 'stimulus_id', 'choice_state') == [
        (condition, repetition, sid, state)
        for condition in ('baseline', 'again') for repetition in (0, 1)
        for sid, state in (('cat', 'valid'), ('boat', 'valid'), ('unknown', 'missing'))]
    reopened = ExperimentResult.open(tmp_path / 'run').trials()
    assert reopened.equals(trials)


@pytest.mark.parametrize('channel, answers, units, columns, expected', [
    ('behavior:rating', {'cat': 6.0, 'boat': 9.0}, None, ('rating', 'rating_state'),
     [(6.0, 'valid'), (9.0, 'invalid'), (None, 'missing')]),
    ('behavior:text', {'cat': 'A cat.', 'boat': 'A boat.'}, None, ('text', 'text_state'),
     [('A cat.', 'valid'), ('A boat.', 'valid'), (None, 'missing')]),
    ('behavior:rt', {'cat': 450.0, 'boat': -1.0}, 'ms', ('rt', 'rt_state'),
     [(450.0, 'valid'), (-1.0, 'invalid'), (None, 'missing')]),
    ('behavior:probabilities', {'cat': np.array([0.8, 0.2]), 'boat': np.array([0.5, 0.6])}, None,
     ('p(animal)', 'probabilities_state'), [(0.8, 'valid'), (0.5, 'invalid'), (None, 'missing')]),
])
def test_every_behavior_format_gives_one_row_per_trial(tmp_path, channel, answers, units, columns, expected):
    response = {'behavior:rating': Scale(1, 7, 1), 'behavior:text': Text()}.get(
        channel, Choice(['animal', 'vehicle'], reaction_time='surprisal') if channel == 'behavior:rt' else CHOICE)
    trials = _experiment(_Cells(channel, answers, units), TaskContext(response=response),
                         tmp_path / 'run', formats=[channel.split(':', 1)[1]]).trials()
    assert _rows(trials, *columns) == expected


def test_repeated_stimuli_give_one_row_per_occurrence(tmp_path):
    stimuli = StimulusSet({'stimulus_id': ['a', 'b', 'a'], 'sentence': ['one', 'two', 'one again']})
    stimuli.identifier = 'trial-states-repeated'
    subject = _Cells('behavior:choice', {('a', 0): 'animal', ('b', 0): 'vehicle', ('a', 1): 'vehicle'})
    trials = _experiment(subject, TaskContext(response=CHOICE), tmp_path / 'run', stimuli=stimuli).trials()
    assert _rows(trials, 'stimulus_id', 'sentence', 'choice') == [
        ('a', 'one', 'animal'), ('b', 'two', 'vehicle'), ('a', 'one again', 'vehicle')]


def test_a_failed_session_falls_back_to_the_observed_events_and_says_so(tmp_path):
    class _AnswersThenFails(_Cells):
        def participate(self, session):
            event = None
            while (event := session.next_input()) is not None:
                if event.meta.get('stimulus_id') == 'cat':
                    session.respond(StreamEvent('behavior:choice', 'animal', event.t_ms, {'stimulus_id': 'cat'}))
                    raise RuntimeError('the model failed')
    with pytest.raises(RuntimeError, match='the model failed'):
        _experiment(_AnswersThenFails('behavior:choice', {}), TaskContext(response=CHOICE), tmp_path / 'run')
    trials = ExperimentResult.open(tmp_path / 'run').trials()
    assert _rows(trials, 'stimulus_id', 'choice', 'rows_from') == [('cat', 'animal', 'observed events')]
    assert 'choice_state' not in trials


# --- a stimulus column of arrays --------------------------------------------------

def test_a_stimulus_column_of_arrays_is_left_out_of_the_presented_trials(tmp_path):
    stimuli = StimulusSet({'stimulus_id': ['cat', 'boat', 'unknown'], 'sentence': ['cat', 'boat', 'ambiguous'],
                           'vector': [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]})
    stimuli.identifier = 'trial-states-vector'
    subject = _Cells('behavior:text', {'cat': 'A cat.', 'boat': 'A boat.', 'unknown': 'Unclear.'})
    result = _experiment(subject, TaskContext(response=Text()), tmp_path / 'run', stimuli=stimuli)
    trials = result.trials()
    assert _rows(trials, 'stimulus_id', 'sentence', 'text', 'text_state') == [
        ('cat', 'cat', 'A cat.', 'valid'), ('boat', 'boat', 'A boat.', 'valid'),
        ('unknown', 'ambiguous', 'Unclear.', 'valid')]
    (answer,) = [row['payload'] for row in RunRecord(tmp_path / 'run' / 'behavior').events()
                 if row['payload']['kind'] == 'answer']
    assert set(answer['payload'].meta['trials']) == {'stimulus_id', 'sentence'}     # no 'vector'


# --- the fallback keeps every observed row ----------------------------------------

def _one_row(stimulus_id, label):
    row = [1.0, 0.0] if label == 'animal' else [0.0, 1.0]
    return BehavioralAssembly(np.array([row], dtype=np.float32), dims=['presentation', 'choice'], coords={
        'stimulus_id': ('presentation', [stimulus_id]), 'choice': ['animal', 'vehicle'],
        'raw_answer': ('presentation', [label])},
        attrs={'response': response_record('behavior:choice', 'test')})


class _AnswersTwoThenFails(Subject):
    """Answers 'cat' and 'boat' (as one-row assemblies, cells, or one of each), then raises."""
    identifier, in_channels, out_channels, required_channels = 'two-then-fails', {'text'}, {'behavior:choice'}, set()

    def __init__(self, forms):
        self.forms = forms

    def participate(self, session):
        answers = {'cat': 'animal', 'boat': 'vehicle'}
        for form, (sid, label) in zip(self.forms, answers.items()):
            session.next_input()
            payload = _one_row(sid, label) if form == 'assembly' else label
            session.respond(StreamEvent('behavior:choice', payload, None, {'stimulus_id': sid}))
        raise RuntimeError('the model failed')

    def reset(self):
        pass


@pytest.mark.parametrize('forms', [('assembly', 'assembly'), ('cell', 'cell'), ('cell', 'assembly')])
def test_a_failed_session_keeps_every_observed_row(tmp_path, forms):
    with pytest.raises(RuntimeError, match='the model failed'):
        _experiment(_AnswersTwoThenFails(forms), TaskContext(response=CHOICE), tmp_path / 'run')
    trials = ExperimentResult.open(tmp_path / 'run').trials()
    assert _rows(trials, 'stimulus_id', 'choice', 'rows_from') == [
        ('cat', 'animal', 'observed events'), ('boat', 'vehicle', 'observed events')]


def test_the_review_case_reopens_as_two_exact_rows(tmp_path):
    """M9-08: a -> x, then b -> y as one-row assemblies, then an error: two rows, not one."""
    with pytest.raises(RuntimeError, match='the model failed'):
        _experiment(_AnswersTwoThenFails(('assembly', 'assembly')), TaskContext(response=CHOICE),
                    tmp_path / 'run')
    trials = ExperimentResult.open(tmp_path / 'run').trials()
    assert _rows(trials, 'stimulus_id', 'choice', 'choice_raw') == [
        ('cat', 'animal', 'animal'), ('boat', 'vehicle', 'vehicle')]


def test_an_older_saved_run_with_several_assembly_events_has_one_row_each(tmp_path):
    """A run saved before finalized answers existed: its behavior record holds only the
    observed events (written here by hand in that form)."""
    import shutil
    from brainscore.run_record import RunRecorder

    class _OneRowEach(Subject):
        identifier, in_channels, out_channels, required_channels = 'one-row-each', {'text'}, {'behavior:choice'}, set()

        def participate(self, session):
            for sid, label in (('cat', 'animal'), ('boat', 'vehicle'), ('unknown', 'animal')):
                session.next_input()
                session.respond(StreamEvent('behavior:choice', _one_row(sid, label), None, {'stimulus_id': sid}))

        def reset(self):
            pass
    _experiment(_OneRowEach(), TaskContext(response=CHOICE), tmp_path / 'new')
    old = tmp_path / 'old'
    old.mkdir()
    shutil.copy(tmp_path / 'new' / 'experiment.json', old / 'experiment.json')
    record = RunRecord(tmp_path / 'new' / 'behavior')
    envelopes = [row['payload'] for row in record.events() if row['payload']['kind'] != 'answer']
    with RunRecorder(old / 'behavior', metadata={'experiment': 'older-run', 'plan': {}}) as recorder:
        for envelope in envelopes:
            recorder.record('input' if envelope['kind'] == 'input' else 'output', envelope)
    trials = ExperimentResult.open(old).trials()
    assert _rows(trials, 'stimulus_id', 'choice', 'rows_from') == [
        ('cat', 'animal', 'observed events'), ('boat', 'vehicle', 'observed events'),
        ('unknown', 'animal', 'observed events')]
