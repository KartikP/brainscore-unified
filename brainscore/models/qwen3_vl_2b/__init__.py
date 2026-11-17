from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('qwen3-vl-2b')


model_registry['qwen3-vl-2b'] = _load
