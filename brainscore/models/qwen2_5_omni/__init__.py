from brainscore import model_registry


def _loader(identifier):
    def load():  # defer heavy imports to load time (keep `import brainscore` light)
        from .model import get_model
        return get_model(identifier)
    return load


model_registry['qwen2.5-omni-7b'] = _loader('qwen2.5-omni-7b')    # Apache-2.0
model_registry['qwen2.5-omni-3b'] = _loader('qwen2.5-omni-3b')    # Qwen Research License (research use)
