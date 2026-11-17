from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('smolvlm2-256m')


model_registry['smolvlm2-256m'] = _load
