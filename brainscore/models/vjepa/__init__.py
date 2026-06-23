from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('vjepa2-vitl')


model_registry['vjepa2-vitl'] = _load
