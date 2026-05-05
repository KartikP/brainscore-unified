from brainscore import model_registry
from .model import get_model

model_registry['vjepa1-wav2vec2'] = lambda: get_model('vjepa1-wav2vec2')
