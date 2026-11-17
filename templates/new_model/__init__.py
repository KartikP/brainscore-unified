"""Registers your model in brainscore.model_registry so load_model('your-model') works.

To install: copy this folder to ``brainscore/models/your_name/``; it registers by
its presence (no other file changes).

``model.py`` imports torch and its wrapper. Import it inside ``_load`` (at load time),
not at the top of this file, so ``import brainscore`` stays light.
"""
from brainscore import model_registry


def _load():
    from .model import get_model
    return get_model()


model_registry['your-model'] = _load
