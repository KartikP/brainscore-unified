from brainscore import model_registry
from .model import get_model

model_registry['vjepa2-vitl'] = lambda: get_model('vjepa2-vitl')
