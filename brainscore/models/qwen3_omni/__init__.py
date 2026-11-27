from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('qwen3-omni-30b-a3b-thinking')


model_registry['qwen3-omni-30b-a3b-thinking'] = _load
