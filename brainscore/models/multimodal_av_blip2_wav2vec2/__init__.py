from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('blip2-wav2vec2')


model_registry['blip2-wav2vec2'] = _load
