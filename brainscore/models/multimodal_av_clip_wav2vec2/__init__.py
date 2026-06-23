from brainscore import model_registry
from ._ids import SUPPORTED_IDENTIFIERS


def _make(_id):  # defer heavy model.py import to load time
    def _load():
        from .model import get_model
        return get_model(_id)
    return _load


for _ident in SUPPORTED_IDENTIFIERS:
    model_registry[_ident] = _make(_ident)
