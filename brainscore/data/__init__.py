"""Unified-owned data plugins.

Modules imported here only register lightweight factories. Actual data I/O
stays deferred until ``brainscore.load_dataset`` or
``brainscore.load_stimulus_set`` is called.
"""

from . import algonauts2025  # noqa: F401
from . import lahner2024  # noqa: F401
