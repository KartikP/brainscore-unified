from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('wav2vec2-base')


model_registry['wav2vec2-base'] = _load
