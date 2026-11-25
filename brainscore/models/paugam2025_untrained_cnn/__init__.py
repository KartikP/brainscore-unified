from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('paugam2025-untrained-cnn')


model_registry['paugam2025-untrained-cnn'] = _load
