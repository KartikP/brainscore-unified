import argparse
import numpy as np
from brainscore.instrumentation import observe
from brainscore.run_record import RunRecorder, RunRecord
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.streaming import StreamEvent
from . import register, LatencyObserver

parser = argparse.ArgumentParser(description='Run an independently installed UMI tool')
parser.add_argument('--out', required=True)
args = parser.parse_args()
register()
model = BrainScoreModel('external-demo', capability_config={'partner-amplitude': True})
tool = LatencyObserver()
with RunRecorder(args.out, metadata={'model': model.identifier, 'fixture': True}) as record:
    with observe(model, tool, record):
        result = model.process(StreamEvent('partner_signal', np.array([3., 4., 0.]), 25.))
assert result.payload == 5. and 'partner_amplitude' in model.out_channels
assert len(tool.seconds) == 1
assert list(RunRecord(args.out).outputs())[0].payload == 5.
model.reset()
assert model.process(StreamEvent('partner_signal', np.array([0., 0., 1.]), 50.)).payload == 1.
assert model.capability_state('partner-amplitude')['calls'] == 1
print('External capability, observer, record/replay and reset: PASS')
