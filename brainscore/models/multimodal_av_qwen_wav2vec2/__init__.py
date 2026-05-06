from brainscore import model_registry
from .model import get_model

model_registry['qwen2.5-vl-wav2vec2'] = lambda: get_model('qwen2.5-vl-wav2vec2')
