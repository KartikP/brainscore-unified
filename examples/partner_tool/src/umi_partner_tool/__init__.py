"""An independent package: no Brain-Score source modifications."""
from dataclasses import dataclass, field
import numpy as np

from brainscore_core.capabilities import Capability, register_capability
from brainscore_core.extensions import CatalogEntry, StreamEvent, register_channel


@dataclass
class LatencyObserver:
    seconds: list = field(default_factory=list)

    def on_result(self, call, result):
        self.seconds.append(call.duration_s)


class Amplitude(Capability):
    identifier = 'partner-amplitude'
    input_channels = {'partner_signal'}
    output_channels = {'partner_amplitude'}

    def enabled_for(self, model):
        return bool(model.capability_config.get(self.identifier))

    def setup(self, model):
        return {'calls': 0}

    def handles(self, model, event, **kwargs):
        return isinstance(event, StreamEvent) and event.channel == 'partner_signal'

    def process(self, model, event, **kwargs):
        model.capability_state(self.identifier)['calls'] += 1
        return StreamEvent('partner_amplitude', float(np.linalg.norm(event.payload)), event.t_ms)


def register():
    """Call once at process initialization, before constructing models."""
    register_channel(CatalogEntry('partner_signal', 'input', 'StreamEvent',
        'Three finite components in normalized fixture units', 'umi_partner_tool',
        owner='partner-example', shape_validator=lambda x: (
            isinstance(x, np.ndarray) and x.shape == (3,) and bool(np.isfinite(x).all()))))
    register_channel(CatalogEntry('partner_amplitude', 'output', 'StreamEvent',
        'Euclidean amplitude in normalized fixture units', 'umi_partner_tool',
        owner='partner-example', shape_validator=lambda x: isinstance(x, float) and bool(np.isfinite(x))))
    register_capability(Amplitude())
