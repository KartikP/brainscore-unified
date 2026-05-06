from brainscore import model_registry
from .model import get_model

model_registry['random-wav2vec2-base'] = lambda: get_model('random-wav2vec2-base')
