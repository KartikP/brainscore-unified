"""Recorded sessions replay with their per-measurement settings."""
from contextlib import contextmanager

from brainscore.experiments import (Experiment, RecordInputsOutputs, SessionProtocol,
                                    compare_outputs, replay_sessions)
from brainscore_core.model_interface import Subject
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly


class _TrialCounter(Subject):
    """Answers with the trial count it was asked for, so a lost setting shows."""
    identifier = 'trial-counter'
    in_channels = {'text'}
    out_channels = {'behavior'}
    required_channels = set()

    def participate(self, session):
        while session.next_input() is not None:
            pass
        trials = (getattr(session, 'settings', None) or {}).get('number_of_trials', 1)
        session.respond(StreamEvent('behavior', BehavioralAssembly([trials], dims=['presentation']), None))


@contextmanager
def _factory(trial):
    session = Session([StreamEvent('text', 'hello', None)], ['behavior'])
    session.settings = {'number_of_trials': 7, 'require_variance': True}
    yield session


def test_replay_keeps_measurement_settings(tmp_path):
    protocol = SessionProtocol('settings', _factory, input_channels=['text'],
                               output_channels=['behavior'])
    first = Experiment(subject=_TrialCounter(), protocol=protocol,
                       tools=[RecordInputsOutputs()], output_dir=tmp_path / 'first').run()
    second = Experiment(subject=_TrialCounter(), protocol=replay_sessions(first.directory),
                        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'second').run()
    assert compare_outputs(first.directory, second.directory)['equal']
