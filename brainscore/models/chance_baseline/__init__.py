from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('chance-baseline')


model_registry['chance-baseline'] = _load
