from brainscore import model_registry
from .model import get_model

model_registry['blip2-wav2vec2'] = lambda: get_model('blip2-wav2vec2')
