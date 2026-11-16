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
        yield stimulus_session(table, ['behavior'], TaskContext(
            task_type='probabilities',
            instruction='Is this an animal?',
            label_set=['yes', 'no'],
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
