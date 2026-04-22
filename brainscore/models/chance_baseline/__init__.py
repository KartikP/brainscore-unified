from brainscore import model_registry
from .model import get_model

model_registry['chance-baseline'] = lambda: get_model('chance-baseline')
