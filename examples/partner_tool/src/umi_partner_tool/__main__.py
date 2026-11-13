import argparse
import numpy as np
from brainscore.instrumentation import observed
from brainscore.run_record import RunRecorder, RunRecord
from brainscore import BrainScoreModel
from brainscore_core.streaming import Session, StreamEvent
from . import register, LatencyObserver


def amplitude(model, signal, t_ms, *sinks):
    """One session: send the signal, return the subject's amplitude response."""
    session = Session([StreamEvent('partner_signal', signal, t_ms)], ('partner_amplitude',))
    with observed(session, _Both(sinks)):
        model.participate(session)
    (response,) = session.emitted
    return response


class _Both:
    def __init__(self, sinks):
        self.sinks = sinks

    def record(self, direction, event):
        for sink in self.sinks:
            sink.record(direction, event)


parser = argparse.ArgumentParser(description='Run an independently installed UMI tool')
parser.add_argument('--out', required=True)
args = parser.parse_args()
register()
model = BrainScoreModel('external-demo', capability_config={'partner-amplitude': True})
tool = LatencyObserver()
with RunRecorder(args.out, metadata={'model': model.identifier, 'fixture': True}) as record:
    result = amplitude(model, np.array([3., 4., 0.]), 25., tool, record)
assert result.payload == 5. and 'partner_amplitude' in model.out_channels
assert len(tool.seconds) == 1
assert list(RunRecord(args.out).outputs())[0].payload == 5.
model.reset()
assert amplitude(model, np.array([0., 0., 1.]), 50.).payload == 1.
assert model.capability_state('partner-amplitude')['calls'] == 1
print('External capability, observer, record/replay and reset: PASS')
