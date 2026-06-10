from brainscore import model_registry
from .model import get_model, REGISTRY

for _identifier in REGISTRY:
    # default arg binds the loop variable per-iteration
    model_registry[_identifier] = lambda _id=_identifier: get_model(_id)
