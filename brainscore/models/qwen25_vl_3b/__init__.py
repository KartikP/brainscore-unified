from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('qwen2.5-vl-3b')


model_registry['qwen2.5-vl-3b'] = _load


def _load_vwfa():
    from .model import get_model, ABLATION_VARIANT
    return get_model(ABLATION_VARIANT)


model_registry['qwen2.5-vl-3b-vwfa'] = _load_vwfa


def _loader(identifier):
    def load():
        from .model import get_model
        return get_model(identifier)
    return load


# The 7B: the same vision tower and parts, a larger decoder.
model_registry['qwen2.5-vl-7b'] = _loader('qwen2.5-vl-7b')
model_registry['qwen2.5-vl-7b-vwfa'] = _loader('qwen2.5-vl-7b-vwfa')

# Explicit precision identity; historical float16 registrations stay unchanged.
model_registry['qwen2.5-vl-3b-bf16'] = _loader('qwen2.5-vl-3b-bf16')
