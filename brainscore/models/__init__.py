"""Unified Brain-Score model registrations.

Each subpackage's __init__.py adds an entry to ``brainscore.model_registry``.
They're imported here so the registry populates on package load.
"""

from . import clip_vit_b_32  # noqa: F401
from . import qwen25_vl_3b   # noqa: F401
from . import blip2_opt_2_7b  # noqa: F401
from . import gpt2            # noqa: F401
from . import random_vit_b_32  # noqa: F401   null-control: untrained ViT + logistic
from . import chance_baseline   # noqa: F401   null-control: uniform probabilities
