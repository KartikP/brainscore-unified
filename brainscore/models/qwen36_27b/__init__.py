"""Qwen3.6-27B vision-language model. Registered lazily; weights load only when called."""

from brainscore import model_registry


def get_model(identifier):  # defer the model.py import to load time
    from .model import get_model
    return get_model(identifier)


model_registry['qwen3.6-27b'] = lambda: get_model('qwen3.6-27b')
