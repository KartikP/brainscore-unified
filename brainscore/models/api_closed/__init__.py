from brainscore import model_registry
from .model import get_model, REGISTRY, _REPLY_SUFFIX

for _identifier in REGISTRY:
    # default arg binds the loop variable per-iteration
    model_registry[_identifier] = lambda _id=_identifier: get_model(_id)
    model_registry[_identifier + _REPLY_SUFFIX] = (
        lambda _id=_identifier + _REPLY_SUFFIX: get_model(_id))
