"""BrainScoreModel capability registry."""

from .base import Capability
from .registry import capability_registry, register_capability

# Import the built-in session parts so they register.
from . import sessions as _sessions  # noqa: F401
