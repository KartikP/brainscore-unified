from brainscore import model_registry


def _make(_id):  # defer heavy model.py import to load time
    def _load():
        from .model import get_model
        return get_model(_id)
    return _load


model_registry['vjepa1-wav2vec2'] = _make('vjepa1-wav2vec2')
model_registry['random-vjepa1-wav2vec2'] = _make('random-vjepa1-wav2vec2')
model_registry['vjepa1-random-wav2vec2'] = _make('vjepa1-random-wav2vec2')
model_registry['random-vjepa1-random-wav2vec2'] = _make('random-vjepa1-random-wav2vec2')
