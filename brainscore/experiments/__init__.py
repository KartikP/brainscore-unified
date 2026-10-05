"""Compose protocols and tools without extending the Subject contract."""
from .runner import Experiment, ExperimentResult, Tool, Trial, SessionProtocol, CallableProtocol
from .tools import RecordInputsOutputs, RecordActivity, Ablate, TorchInstrumentation, ObserveCalls

__all__ = ['Experiment', 'ExperimentResult', 'Tool', 'SessionProtocol',
           'CallableProtocol', 'Trial', 'ObserveCalls', 'RecordInputsOutputs', 'RecordActivity', 'Ablate',
           'TorchInstrumentation']
from .replay import replay_sessions, replay_calls, compare_outputs, read_events
__all__ += ['replay_sessions', 'replay_calls', 'compare_outputs', 'read_events']
