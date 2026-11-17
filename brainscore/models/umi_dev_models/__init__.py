from brainscore import model_registry


def _loader(identifier):
    def load():  # heavy imports at load time, not at `import brainscore`
        from .model import get_model
        return get_model(identifier)
    return load


model_registry['qwen3-vl-2b_UMI-dev'] = _loader('qwen3-vl-2b_UMI-dev')
model_registry['qwen3-vl-4b_UMI-dev'] = _loader('qwen3-vl-4b_UMI-dev')
model_registry['qwen3-vl-8b_UMI-dev'] = _loader('qwen3-vl-8b_UMI-dev')
model_registry['qwen3-vl-32b_UMI-dev'] = _loader('qwen3-vl-32b_UMI-dev')
model_registry['qwen3-vl-30b-a3b_UMI-dev'] = _loader('qwen3-vl-30b-a3b_UMI-dev')
model_registry['qwen2.5-vl-3b_UMI-dev'] = _loader('qwen2.5-vl-3b_UMI-dev')
model_registry['qwen2.5-vl-7b_UMI-dev'] = _loader('qwen2.5-vl-7b_UMI-dev')
model_registry['smolvlm2-256m_UMI-dev'] = _loader('smolvlm2-256m_UMI-dev')
model_registry['smolvlm2-2.2b_UMI-dev'] = _loader('smolvlm2-2.2b_UMI-dev')
model_registry['gemma-3-4b_UMI-dev'] = _loader('gemma-3-4b_UMI-dev')
model_registry['gemma-3-12b_UMI-dev'] = _loader('gemma-3-12b_UMI-dev')
model_registry['gemma-3-27b_UMI-dev'] = _loader('gemma-3-27b_UMI-dev')
model_registry['internvl3-1b_UMI-dev'] = _loader('internvl3-1b_UMI-dev')
model_registry['internvl3-8b_UMI-dev'] = _loader('internvl3-8b_UMI-dev')
model_registry['internvl3-14b_UMI-dev'] = _loader('internvl3-14b_UMI-dev')
