"""Record and replay built-in sessions through the real BrainScoreModel driver."""
from contextlib import contextmanager

import numpy as np
import pytest
import xarray as xr

from brainscore.experiments import (
    Experiment, RecordInputsOutputs, SessionProtocol, compare_outputs,
    replay_sessions,
)
from brainscore.experiments.replay import _equal, read_events
from brainscore import BrainScoreModel
from brainscore_core.contract import Subject, TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import (
    stimulus_session,
)
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


def stimuli():
    result = StimulusSet({
        'stimulus_id': ['short', 'long'],
        'sentence': ['cat', 'elephant'],
        'condition': ['easy', 'hard'],
    })
    result.identifier = 'replay-fixture'
    return result


def extract(stimuli, layers=None):
    return NeuroidAssembly(
        np.array([[len(text)] * len(layers) for text in stimuli['sentence']], dtype=float),
        dims=['presentation', 'neuroid'],
        coords={
            'stimulus_id': ('presentation', stimuli['stimulus_id'].tolist()),
            'condition': ('presentation', stimuli['condition'].tolist()),
            'neuroid_id': ('neuroid', [f'{layer}.0' for layer in layers]),
            'layer': ('neuroid', layers),
        },
    )


extract.identifier = 'length-extractor'


def run(subject, factory, directory, outputs):
    return Experiment(
        subject=subject,
        protocol=SessionProtocol(
            identifier='native-session',
            session_factory=factory,
            required_input_channels=['text'],
            requested_output_channels=outputs,
        ),
        tools=[RecordInputsOutputs()],
        output_dir=directory,
    ).run()


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('regions', ['IT', ['V4', 'IT']])
def test_neural_session_collection_and_replay_preserve_metadata(tmp_path, streaming, regions):
    subject = BrainScoreModel(
        identifier='length',
        model=None,
        extractor={'text': extract},
        region_layer_map={'IT': 'length', 'V4': 'early_length'},
    )

    channels = ('neural:IT',) if isinstance(regions, str) else ('neural:V4', 'neural:IT')

    @contextmanager
    def factory(trial):
        yield stimulus_session(stimuli(), channels, time_bins=[(70, 170)], streaming=streaming)

    baseline = run(subject, factory, tmp_path / 'baseline', channels)
    assembly = baseline.value[0] if isinstance(regions, str) else baseline.value[0]['neural:IT']
    assert isinstance(assembly, NeuroidAssembly)
    assert assembly['condition'].values.tolist() == ['easy', 'hard']
    replay = Experiment(
        subject=subject,
        protocol=replay_sessions(directory=baseline.directory),
        tools=[RecordInputsOutputs()],
        output_dir=tmp_path / 'replay',
    ).run()
    assert compare_outputs(baseline.directory, replay.directory)['equal']
    context = next(e['payload'] for e in read_events(replay.directory)
                   if e['kind'] == 'session_context')
    assert context['time_bins'] == [(70, 170)]
    assert context['outputs'] == channels


def test_behavior_session_collection_and_replay_preserve_task(tmp_path):
    calls = []

    def generate(stimulus_row, instruction, label_set):
        calls.append((stimulus_row['stimulus_id'], instruction, tuple(label_set)))
        return 'yes'

    subject = BrainScoreModel(
        identifier='answers',
        model=None,
        in_channels={'text'},
        region_layer_map={},
        answer=generate,
    )

    @contextmanager
    def factory(trial):
        table = stimuli()
        from brainscore_core.responses import Choice
        yield stimulus_session(table, ['behavior'], TaskContext(
            response=Choice(['yes', 'no']),
            instruction='Is this an animal?',
            metadata={'stimulus_set': table},
        ))

    baseline = run(subject, factory, tmp_path / 'baseline', ['behavior'])
    assert baseline.value[0].shape[0] == 2
    original_calls = list(calls)
    replay = Experiment(
        subject=subject,
        protocol=replay_sessions(directory=baseline.directory),
        tools=[RecordInputsOutputs()],
        output_dir=tmp_path / 'replay',
    ).run()
    assert calls == original_calls * 2
    assert compare_outputs(baseline.directory, replay.directory)['equal']


def test_assembly_comparison_checks_labels_values_and_attributes():
    original = xr.DataArray([1., np.nan], dims=['unit'], coords={'unit': ['a', 'b']})
    assert _equal(original, original.copy(deep=True))
    assert not _equal(original, original.assign_coords(unit=['b', 'a']))
    assert not _equal(original, original.fillna(0))
    assert not _equal(original, original.assign_attrs(units='mV'))
    assert not _equal(original, original.astype(np.float32))


def test_neural_replay_with_generic_collector_restores_each_trials_request(tmp_path):
    """Exercise replay independently of built-in collect() defaults."""
    subject = BrainScoreModel(
        identifier='length',
        model=None,
        extractor={'text': extract},
        region_layer_map={'IT': 'length', 'V4': 'early_length'},
    )

    @contextmanager
    def factory(trial):
        region = trial.identifier
        table = stimuli()
        session = Session([
            StreamEvent(channel='text', payload=row['sentence'], t_ms=index)
            for index, row in table.iterrows()
        ])
        session.stimuli = table
        session.time_bins = [(70, 170)]
        session.outputs = (f'neural:{region}',)
        yield session

    baseline = Experiment(
        subject=subject,
        protocol=SessionProtocol(
            identifier='different-region-per-trial',
            session_factory=factory,
            repetitions=['V4', 'IT'],
            required_input_channels=['text'],
            requested_output_channels=['neural:V4', 'neural:IT'],
        ),
        tools=[RecordInputsOutputs()],
        output_dir=tmp_path / 'baseline',
    ).run()
    replay = Experiment(
        subject=subject,
        protocol=replay_sessions(directory=baseline.directory),
        tools=[RecordInputsOutputs()],
        output_dir=tmp_path / 'replay',
    ).run()
    assert compare_outputs(baseline.directory, replay.directory)['equal']


class JointReader(Subject):
    """A stand-in subject that answers each word and records a region with it."""
    identifier = 'joint-reader'
    in_channels = {'text'}
    out_channels = {'behavior:choice', 'neural:A'}

    def participate(self, session):
        while (event := session.next_input()) is not None:
            if event.channel != 'text':
                continue
            meta = {'stimulus_id': event.meta['stimulus_id']}
            session.respond(StreamEvent('behavior:choice', 'yes' if len(event.payload) > 3 else 'no',
                                        event.t_ms, meta))
            session.respond(StreamEvent('neural:A', np.array([float(len(event.payload)), 1.0]),
                                        event.t_ms, meta))



def test_a_recorded_joint_session_replays_to_equal_outputs(tmp_path):
    """A record of a session that requests an answer and a region replays, and the replayed
    outputs compare as equal to the recorded ones."""
    subject = JointReader()
    channels = ('behavior:choice', 'neural:A')
    task = TaskContext(instruction='Long word?', response=Choice(['yes', 'no']))

    @contextmanager
    def factory(trial):
        yield stimulus_session(stimuli(), channels, task)

    recorded = run(subject, factory, tmp_path / 'recorded', channels)
    replay = Experiment(subject=subject, protocol=replay_sessions(recorded.directory),
                        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'replay').run()
    assert compare_outputs(recorded.directory, replay.directory)['equal']


def test_a_replay_of_a_record_without_an_output_request_gives_the_same_trial_table(tmp_path):
    """A session that requested no outputs has no finalized answer; its replay has none
    either, so both trial tables are built from the answer events and are equal."""
    from brainscore.experiments import RecordBehavior

    class Namer(Subject):
        identifier = 'namer'
        in_channels = {'text'}
        out_channels = {'behavior:choice'}

        def participate(self, session):
            while (event := session.next_input()) is not None:
                if event.channel == 'text':
                    session.respond(StreamEvent('behavior:choice', 'cat', event.t_ms,
                                                {'stimulus_id': event.meta['stimulus_id']}))

    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))

    @contextmanager
    def factory(trial):
        yield Session([StreamEvent('text', 'a cat', 125.0, {'stimulus_id': 'a'})], None, task)

    protocol = SessionProtocol('no-request', factory, required_input_channels=['text'],
                               requested_output_channels=['behavior:choice'])
    tools = lambda: [RecordInputsOutputs(), RecordBehavior(['choice'])]
    recorded = Experiment(subject=Namer(), protocol=protocol, tools=tools(),
                          output_dir=tmp_path / 'recorded').run()
    replay = Experiment(subject=Namer(), protocol=replay_sessions(recorded.directory), tools=tools(),
                        output_dir=tmp_path / 'replay').run()
    assert recorded.trials().equals(replay.trials())


@pytest.mark.parametrize('channel, response, number', [('behavior:rating', 'scale', 2.0),
                                                       ('behavior:rt', 'rt', 412.0)])
def test_a_saved_numeric_answer_reads_back_as_the_live_result(tmp_path, channel, response, number):
    """A session that requested no outputs gives its events as the result; a number
    answered as a cell reads back from the saved result as it was kept."""
    from brainscore.experiments.runner import ExperimentResult
    from brainscore_core.responses import Scale

    task = TaskContext(response=Scale(0, 5) if response == 'scale'
                       else Choice(['a', 'b'], reaction_time='surprisal'))

    class Numbers(Subject):
        identifier = 'numbers'
        in_channels = {'text'}
        out_channels = {channel}

        def reset(self):
            pass

        def participate(self, session):
            while (event := session.next_input()) is not None:
                if event.channel == 'text':
                    session.respond(StreamEvent(channel, number, event.t_ms,
                                                {'stimulus_id': 'trial', 'units': 'ms'}))

    @contextmanager
    def factory(repetition):
        yield Session([StreamEvent('text', 'prompt', 1.0, {'stimulus_id': 'trial'})], None, task)

    live = Experiment(subject=Numbers(), protocol=SessionProtocol(
        'numbers', factory, required_input_channels={'text'}, requested_output_channels={channel}),
        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'run').run()
    (kept,), = live.value
    (read,), = ExperimentResult.open(live.directory).value
    assert read.payload == kept.payload
    assert {key: type(value) for key, value in read.payload.items()} == \
        {key: type(value) for key, value in kept.payload.items()}


@pytest.mark.parametrize('options', [[float('nan'), 'b'],
                                     [np.datetime64('2020-01-01'), np.datetime64('2020-01-02')],
                                     [np.timedelta64(1, 'D'), np.timedelta64(2, 'D')], [1 + 2j, 2 + 3j]])
def test_an_answer_reads_back_from_a_record_with_the_options_the_table_holds(tmp_path, options):
    import json
    from brainscore.run_record import PayloadCodec

    session = Session([], ['behavior:probabilities'], TaskContext(response=Choice(options)))
    session.respond(StreamEvent('behavior:probabilities', np.array([0.2, 0.8]), None, {'stimulus_id': 's'}))
    codec = PayloadCodec(tmp_path)
    (read,) = codec.decode(json.loads(json.dumps(codec.encode(list(session.emitted)))))
    held = session.collect('behavior:probabilities')['choice'].values
    assert read.payload['choice'].dtype == held.dtype
    assert [str(option) for option in read.payload['choice']] == [str(option) for option in held]


@pytest.mark.parametrize('stimulus_id', [np.int8(3), np.float32(1.5), np.datetime64('2020-01-01T00:00:00.000000000')])
def test_answers_saved_and_read_back_collect_as_the_live_answers(tmp_path, stimulus_id):
    import json
    from brainscore.run_record import PayloadCodec
    from brainscore_core.responses import Text

    def session():
        return Session([], ['behavior:text'], TaskContext(response=Text()))
    live = session()
    live.respond(StreamEvent('behavior:text', 'hello', None, {'stimulus_id': stimulus_id}))
    live.respond(StreamEvent('behavior:text', 'again', None, {'stimulus_id': 's1'}))
    codec = PayloadCodec(tmp_path)
    read = session()
    read.emitted.extend(codec.decode(json.loads(json.dumps(codec.encode(list(live.emitted))))))
    assert read.collect('behavior:text').identical(live.collect('behavior:text'))



class _CarryingSubject(Subject):
    """Answers probabilities and carries one assembly with an array attribute in the meta."""
    identifier = 'carrying'
    in_channels = {'text'}
    out_channels = {'behavior:probabilities'}

    def __init__(self, nested):
        self.nested = nested

    def participate(self, session):
        from brainscore_core.response_formats import response_record
        from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
        session.declare_inputs(self.in_channels)
        while session.next_input() is not None:
            pass
        ids = list(session.stimuli['stimulus_id'])
        answer = BehavioralAssembly([[0.25, 0.75]] * len(ids), dims=['presentation', 'choice'],
                                    coords={'stimulus_id': ('presentation', ids), 'choice': ['cat', 'dog']},
                                    attrs={'response': response_record('behavior:probabilities', 'external')})
        carried = answer.copy()
        carried.attrs = {'nested': {'values': np.array(self.nested)}}
        session.respond(StreamEvent('behavior:probabilities', answer, None, meta={'carried': carried}))

    def reset(self):
        pass


def _record_answer(directory, nested):
    @contextmanager
    def factory(repetition):
        yield stimulus_session(stimuli(), ['behavior:probabilities'], TaskContext(response=Choice(['cat', 'dog'])))
    protocol = SessionProtocol('carrying', factory, required_input_channels=['text'],
                               requested_output_channels=['behavior:probabilities'])
    return Experiment(subject=_CarryingSubject(nested), protocol=protocol, tools=[RecordInputsOutputs()],
                      output_dir=directory).run().directory


def test_outputs_whose_meta_holds_an_assembly_with_array_attributes_compare(tmp_path):
    """An assembly in event meta whose attributes hold arrays compares; a changed array differs."""
    a, b, c = (_record_answer(tmp_path / name, nested)
               for name, nested in (('a', [1., np.nan]), ('b', [1., np.nan]), ('c', [1., 2.])))
    assert compare_outputs(a, b)['equal']
    assert not compare_outputs(a, c)['equal']
