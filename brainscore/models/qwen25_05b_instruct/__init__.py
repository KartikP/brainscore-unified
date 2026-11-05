"""Small local instruction-following subject, loaded lazily."""
from brainscore import model_registry


def _load():
    from .model import get_model
    return get_model()


model_registry['qwen2.5-0.5b-instruct'] = _load
