"""Random-init Wav2Vec2-base — null control for audio multimodal scoring.

Same architecture as Wav2Vec2-base from facebook/wav2vec2-base, but
weights are initialized from the model's default constructor (random)
with a fixed seed instead of loading the pretrained checkpoint. Pairs
with the existing ``random-vit-b-32`` for vision so multimodal
scoring has a complete null floor across both towers.

Use case: any multimodal benchmark should score this against the same
candidate model as a sanity check. If banded ridge can extract
positive signal from random-init Wav2Vec2 features, that's evidence
the framework is overfitting via α-tuning rather than detecting real
audio contribution.
"""

import numpy as np
import torch

from brainscore_core.model_interface import BrainScoreModel


AUDIO_HF_ID = 'facebook/wav2vec2-base'
AUDIO_LAYER = 'encoder.layers.6'

REGION_LAYER_MAP = {'A1': AUDIO_LAYER}
REGION_MODALITY_MAP = {'A1': 'audio'}


def _build_random_wav2vec2(identifier: str):
    from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2FeatureExtractor
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    from scipy.io import wavfile

    # Load config from HF (architecture only), then construct model
    # from config so weights are randomly initialized rather than
    # loaded from the pretrained checkpoint.
    config = Wav2Vec2Config.from_pretrained(AUDIO_HF_ID)
    torch.manual_seed(0)
    np.random.seed(0)
    model = Wav2Vec2Model(config).eval()

    processor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_HF_ID)

    def _wav_loader(audio_path: str, target_rate: int) -> np.ndarray:
        sr, wav = wavfile.read(audio_path)
        if sr != target_rate:
            raise ValueError(
                f"WAV at {audio_path} has sr={sr}, expected {target_rate}.")
        if wav.dtype.kind == 'i':
            wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
        else:
            wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        return wav

    wrapper = AudioWrapper(
        model=model,
        processor=processor,
        identifier=f'{identifier}-audio',
        backbone_id=identifier,
        layer_aggregation='time_series',
        max_duration_sec=60.0,
        batch_size=4,
        audio_loader=_wav_loader,
        audio_input_key='input_values',
    )
    return wrapper


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'random-wav2vec2-base'
    audio_wrapper = _build_random_wav2vec2(identifier)
    return BrainScoreModel(
        identifier=identifier,
        model=None,
        region_layer_map=REGION_LAYER_MAP,
        region_modality_map=REGION_MODALITY_MAP,
        preprocessors={'audio': audio_wrapper},
        required_modalities={'audio'},
        activations_model=None,
        visual_degrees=8,
    )
