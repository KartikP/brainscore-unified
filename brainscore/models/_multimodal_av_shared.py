"""Shared audio-tower factory for the multimodal A+V model registrations.

CLIP, BLIP-2, Qwen-VL all pair with the same Wav2Vec2-base audio
backbone. This module factors out the audio-side wrapper construction
(plus the random-init null variant) so each VLM combo's model.py only
has to handle the visual side.
"""
from typing import Tuple

import numpy as np


AUDIO_HF_ID = 'facebook/wav2vec2-base'
AUDIO_LAYER = 'encoder.layers.6'


def build_audio_wrapper(combo_identifier: str, random_init: bool = False):
    """Wav2Vec2-base AudioWrapper. Identical contract to the
    multimodal_av_vjepa_wav2vec2 audio path; consolidated here so all
    VLM+audio combos share a single source of truth."""
    from transformers import (
        Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2FeatureExtractor)
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    from scipy.io import wavfile
    import torch

    if random_init:
        config = Wav2Vec2Config.from_pretrained(AUDIO_HF_ID)
        torch.manual_seed(0)
        model = Wav2Vec2Model(config).eval()
    else:
        model = Wav2Vec2Model.from_pretrained(AUDIO_HF_ID).eval()
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

    backbone_id = 'random-wav2vec2-base' if random_init else 'wav2vec2-base'
    wrapper = AudioWrapper(
        model=model,
        processor=processor,
        identifier=f'{combo_identifier}-audio',
        backbone_id=backbone_id,
        layer_aggregation='time_series',
        max_duration_sec=60.0,
        batch_size=4,
        audio_loader=_wav_loader,
        audio_input_key='input_values',
    )
    return wrapper, model
