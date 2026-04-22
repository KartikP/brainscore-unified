from brainscore import model_registry
from .model import get_model

model_registry['gpt2'] = lambda: get_model('gpt2')
