"""Wav2Vec2-base (facebook/wav2vec2-base), trained, as an audio-only subject: the audio part of the
multimodal_av models (``brainscore.models._shared.build_audio_wrapper``), alone.

Input: ``audio`` only. Regions: ``A1`` (encoder.layers.6, as the multimodal_av models) and ``cortex``: the 12
transformer layers of the encoder, one band each. Its floor is ``random-wav2vec2-base`` (the same architecture
with random weights).
"""
from brainscore import BrainScoreModel
from brainscore.models._shared import AUDIO_LAYER, build_audio_wrapper

CORTEX_LAYERS = tuple(f'encoder.layers.{i}' for i in range(12))


def region_layers() -> dict:
    from brainscore.selection import CompositeSelector
    return {'A1': AUDIO_LAYER, 'cortex': CompositeSelector(tuple((layer, None) for layer in CORTEX_LAYERS))}


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'wav2vec2-base'
    audio_wrapper, _ = build_audio_wrapper(identifier)
    return BrainScoreModel(identifier=identifier, region_layer_map=region_layers(), extractor={'audio': audio_wrapper})
