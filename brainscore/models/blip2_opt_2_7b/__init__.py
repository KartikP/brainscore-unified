from brainscore import model_registry
from .model import get_model

model_registry['blip2-opt-2.7b'] = lambda: get_model('blip2-opt-2.7b')
