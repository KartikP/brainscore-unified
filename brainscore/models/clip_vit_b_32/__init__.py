from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('clip-vit-b-32')


model_registry['clip-vit-b-32'] = _load
