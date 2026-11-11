"""Deprecated module name; use ``brainscore.capabilities.closed_loop``."""
from .closed_loop import ClosedLoopCapability

# ponytail: alias kept for one release, delete with the DROID aliases in events.py.
EmbodiedActionCapability = ClosedLoopCapability
