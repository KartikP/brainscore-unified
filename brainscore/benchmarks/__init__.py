"""Unified Brain-Score benchmarks.

Benchmarks here call `model.process()` directly (no legacy look_at/digest_text
shims) and require models that implement the `UnifiedModel` interface.
"""

# Import each benchmark subpackage so its __init__.py runs and registers
# entries in `brainscore.benchmark_registry`.
from . import roar_yeatman2021  # noqa: F401
from . import lahner2024        # noqa: F401  naturalistic fMRI (scaffolding)
