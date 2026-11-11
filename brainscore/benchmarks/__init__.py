"""Unified Brain-Score benchmarks.

Benchmarks here reach a subject only through a session (a core helper such as
`neural_response`, which calls `participate`) and declare what they send
(`required_input_channels`), what they ask for (`requested_output_channels`)
and how their trials relate (`trial_protocol`).
"""

# Import each benchmark subpackage so its __init__.py runs and registers
# entries in `brainscore.benchmark_registry`.
from . import roar_yeatman2021  # noqa: F401
from . import induced_dyslexia  # noqa: F401  perturbation benchmark (ablation)
from . import grid_game         # noqa: F401  closed-loop benchmark
from . import lahner2024        # noqa: F401  naturalistic fMRI (scaffolding)
from . import algonauts2025     # noqa: F401  multimodal movie fMRI (scaffold; data on EC2)
from . import lebel2023         # noqa: F401  whole-cortex story-listening fMRI
from . import coggym            # noqa: F401  pinned external cognitive experiments

from . import ewok  # noqa: F401
