"""Qwen3.6-27B vision-language model. Registered lazily; weights load only when called."""

from brainscore import model_registry

from .model import get_model

model_registry['qwen3.6-27b'] = lambda: get_model('qwen3.6-27b')
