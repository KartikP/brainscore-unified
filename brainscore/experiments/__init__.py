"""Compose protocols and tools without extending the Subject contract."""
from .runner import Experiment, ExperimentResult, Tool, Trial, SessionProtocol, CallableProtocol
from .tools import RecordInputsOutputs, RecordActivity, Ablate, ScaleActivity, TorchInstrumentation, ObserveCalls

__all__ = ['Experiment', 'ExperimentResult', 'Tool', 'SessionProtocol',
           'CallableProtocol', 'Trial', 'ObserveCalls', 'RecordInputsOutputs', 'RecordActivity', 'Ablate',
           'TorchInstrumentation', 'ScaleActivity']
from .replay import replay_sessions, replay_calls, compare_outputs, read_events
__all__ += ['replay_sessions', 'replay_calls', 'compare_outputs', 'read_events']
from .reasoning import RecordReasoning
__all__ += ['RecordReasoning']
from .backends.openpi import OpenPIInstrumentation
from .backends.openpi_remote import RemoteOpenPIInstrumentation, OpenPIPolicyClient, OpenPIToolServer
__all__ += ['OpenPIInstrumentation', 'RemoteOpenPIInstrumentation',
            'OpenPIPolicyClient', 'OpenPIToolServer']
