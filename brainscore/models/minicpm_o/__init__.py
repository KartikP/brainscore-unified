from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('minicpm-o-4.5')


model_registry['minicpm-o-4.5'] = _load    # Apache-2.0
