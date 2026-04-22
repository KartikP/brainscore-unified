from brainscore import model_registry
from .model import get_model

model_registry['random-vit-b-32'] = lambda: get_model('random-vit-b-32')
