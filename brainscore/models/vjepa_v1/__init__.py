from brainscore import model_registry
from .model import get_model

model_registry['vjepa1-vitl'] = lambda: get_model('vjepa1-vitl')
