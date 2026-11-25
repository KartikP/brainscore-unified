from brainscore import model_registry


def _make(identifier):
    def load():  # defer heavy imports to load time (keep `import brainscore` light)
        from .model import get_model
        return get_model(identifier)
    return load


model_registry['mario-ppo'] = _make('mario-ppo')
model_registry['mario-ppo-untrained'] = _make('mario-ppo-untrained')
