from brainscore import model_registry


def _make(_id):  # defer heavy model.py import to load time
    def _load():
        from .model import get_model
        return get_model(_id)
    return _load


model_registry['clip-wav2vec2'] = _make('clip-wav2vec2')
model_registry['random-clip-wav2vec2'] = _make('random-clip-wav2vec2')
model_registry['clip-random-wav2vec2'] = _make('clip-random-wav2vec2')
model_registry['random-clip-random-wav2vec2'] = _make('random-clip-random-wav2vec2')
