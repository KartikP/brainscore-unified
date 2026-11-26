"""Random-init Wav2Vec2-base — null control for audio multimodal scoring.

Same architecture as Wav2Vec2-base from facebook/wav2vec2-base, but
weights are initialized from the model's default constructor (random)
with a fixed seed instead of loading the pretrained checkpoint. Pairs
with the existing ``random-vit-b-32`` for vision so multimodal
scoring has a complete null floor across both towers.

Use case: any multimodal benchmark should score this against the same
subject as a sanity check. If banded ridge can extract
positive signal from random-init Wav2Vec2 features, that's evidence
the framework is overfitting via α-tuning rather than detecting real
audio contribution.
"""

from brainscore import BrainScoreModel
from brainscore.models._shared import AUDIO_LAYER, build_audio_wrapper

REGION_LAYER_MAP = {'A1': AUDIO_LAYER}
# ``cortex`` (the whole-cortex benchmarks): the 12 encoder layers, one band each, as wav2vec2-base.
CORTEX_LAYERS = tuple(f'encoder.layers.{i}' for i in range(12))


def region_layers() -> dict:
    from brainscore.selection import CompositeSelector
    return {**REGION_LAYER_MAP, 'cortex': CompositeSelector(tuple((layer, None) for layer in CORTEX_LAYERS))}


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'random-wav2vec2-base'
    audio_wrapper, _ = build_audio_wrapper(identifier, random_init=True)
    return BrainScoreModel(
        identifier=identifier,
        region_layer_map=region_layers(),
        extractor={
            'audio': audio_wrapper,
        },
    )
