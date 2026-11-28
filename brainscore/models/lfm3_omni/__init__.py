from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('lfm3-omni-2b')


model_registry['lfm3-omni-2b'] = _load    # LFM Open License v1.0 (research use)
