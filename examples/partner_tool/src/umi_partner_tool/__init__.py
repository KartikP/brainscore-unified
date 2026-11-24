"""An independent package: no Brain-Score source modifications."""
from dataclasses import dataclass, field
import time

import numpy as np

from brainscore.capabilities import Capability
from brainscore_core.io_catalog import CatalogEntry, register_channel
from brainscore_core.streaming import StreamEvent


@dataclass
class LatencyObserver:
    """A session sink for ``observed(session, sink)``: seconds from each input to its output."""
    seconds: list = field(default_factory=list)
    _started: float = 0.0

    def record(self, direction, event):
        if direction == 'input':
            self._started = time.perf_counter()
        else:
            self.seconds.append(time.perf_counter() - self._started)


class Amplitude(Capability):
    """A session capability: it answers each signal with its amplitude."""
    identifier = 'partner-amplitude'
    input_channels = {'partner_signal'}
    output_channels = {'partner_amplitude'}

    def __init__(self):
        self.calls = 0          # this instance's state: give each model its own instance

    def reset(self, subject):
        self.calls = 0

    def supports_session(self, subject, channels):
        return set(channels) == self.output_channels

    def participate(self, subject, session):
        # The session checks each signal and each answer against the catalog.
        while (event := session.next_input()) is not None:
            if event.channel == 'partner_signal':
                self.calls += 1
                session.respond(StreamEvent('partner_amplitude',
                                            float(np.linalg.norm(event.payload)), event.t_ms))


def register():
    """Add the two channels. Call once at process initialization, before constructing models."""
    register_channel(CatalogEntry('partner_signal', 'input', 'StreamEvent',
        'Three finite components in normalized fixture units', 'umi_partner_tool',
        owner='partner-example', shape_validator=lambda x: (
            isinstance(x, np.ndarray) and x.shape == (3,) and bool(np.isfinite(x).all()))))
    register_channel(CatalogEntry('partner_amplitude', 'output', 'StreamEvent',
        'Euclidean amplitude in normalized fixture units', 'umi_partner_tool',
        owner='partner-example', shape_validator=lambda x: isinstance(x, float) and bool(np.isfinite(x))))
