"""BrainScoreModel capability registry."""

from .base import Capability
from .registry import capability_registry, register_capability

# Import the built-in capabilities so they register.
from . import built_in as _built_in  # noqa: F401
