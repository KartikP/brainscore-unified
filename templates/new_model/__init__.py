"""Registers your model in brainscore.model_registry so load_model('your-model') works.

To install: copy this folder to ``brainscore/models/your_name/``, then add
``from . import your_name`` to ``brainscore/models/__init__.py``.
"""
from brainscore import model_registry
from .model import get_model

model_registry['your-model'] = get_model
