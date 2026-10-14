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
from brainscore_core.model_interface import BrainScoreModel, TaskContext
from brainscore_core.streaming import InMemorySession, StreamEvent
from brainscore_core.streaming_helpers import (
    BehavioralSession, StimulusSetSession, StreamingStimulusSetSession,
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
            input_channels=['text'],
            output_channels=outputs,
        ),
        tools=[RecordInputsOutputs()],
        output_dir=directory,
    ).run()


@pytest.mark.parametrize('session_type', [StimulusSetSession, StreamingStimulusSetSession])
@pytest.mark.parametrize('regions', ['IT', ['V4', 'IT']])
def test_neural_session_collection_and_replay_preserve_metadata(tmp_path, session_type, regions):
    subject = BrainScoreModel(
        identifier='length',
        model=None,
        preprocessors={'text': extract},
        region_layer_map={'IT': 'length', 'V4': 'early_length'},
    )

    @contextmanager
    def factory(trial):
        yield session_type(stimulus_set=stimuli(), record=regions, time_bins=[(70, 170)])

    channels = ('neural:IT',) if isinstance(regions, str) else ('neural:V4', 'neural:IT')
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
    assert context['requested_output_channels'] == channels


def test_behavior_session_collection_and_replay_preserve_task(tmp_path):
    calls = []

    def generate(stimulus_row, instruction, label_set):
        calls.append((stimulus_row['stimulus_id'], instruction, tuple(label_set)))
        return 'yes'

    subject = BrainScoreModel(
        identifier='answers',
        model=None,
        preprocessors={'text': lambda inputs: inputs},
        region_layer_map={},
        generation_fn=generate,
    )

    @contextmanager
    def factory(trial):
        yield BehavioralSession(task_context=TaskContext(
            task_type='probabilities',
            instruction='Is this an animal?',
            label_set=['yes', 'no'],
            metadata={'stimulus_set': stimuli()},
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


def test_neural_replay_without_a_collector_restores_each_trials_request(tmp_path):
    """Exercise replay independently of built-in collect() defaults."""
    subject = BrainScoreModel(
        identifier='length',
        model=None,
        preprocessors={'text': extract},
        region_layer_map={'IT': 'length', 'V4': 'early_length'},
    )

    @contextmanager
    def factory(region):
        table = stimuli()
        session = InMemorySession([
            StreamEvent(channel='text', payload=row['sentence'], t_ms=index)
            for index, row in table.iterrows()
        ])
        session.stimulus_set = table
        session.time_bins = [(70, 170)]
        session.requested_output_channels = (f'neural:{region}',)
        yield session

    baseline = Experiment(
        subject=subject,
        protocol=SessionProtocol(
            identifier='different-region-per-trial',
            session_factory=factory,
            trials=['V4', 'IT'],
            input_channels=['text'],
            output_channels=['neural:V4', 'neural:IT'],
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
