from brainscore import model_registry
from .model import get_model, SUPPORTED_IDENTIFIERS

for _ident in SUPPORTED_IDENTIFIERS:
    model_registry[_ident] = (lambda i=_ident: get_model(i))
