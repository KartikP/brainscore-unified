from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('blip2-opt-2.7b')


model_registry['blip2-opt-2.7b'] = _load
