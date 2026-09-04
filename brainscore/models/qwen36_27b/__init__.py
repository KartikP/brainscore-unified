"""Qwen3.6-27B text tower. Registered lazily; weights load only when called."""

from brainscore import model_registry

from .model import get_model

model_registry['qwen3.6-27b'] = lambda: get_model('qwen3.6-27b')
