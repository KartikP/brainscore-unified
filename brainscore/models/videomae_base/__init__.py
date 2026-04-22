from brainscore import model_registry
from .model import get_model

model_registry['videomae-base'] = lambda: get_model('videomae-base')
