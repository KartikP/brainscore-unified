"""V-JEPA v1 (video) + Wav2Vec2-base (audio): the first registered
multimodal Brain-Score model.

Built specifically to exercise the M12-full unified-interface APIs:
- Two preprocessors (`video` + `audio`), one BrainScoreModel
- ``region_modality_map`` routing IT to the video tower and A1 to the
  audio tower in cross-tower extraction
- Multi-modality scoring against a multimodal benchmark
  (``Lahner2024-fMRI-naturalistic-multimodal``)

The two backbones are independent — there is NO fusion network. Each
tower extracts features in parallel; the benchmark handles concatenation.
This matches the convention used by TRIBEv2 and similar A/V probes.

Reuses the V-JEPA v1 ViT-L backbone vendored under
``brainscore.models.vjepa_v1.backbone`` (no second download).
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn

from brainscore_core.model_interface import BrainScoreModel


# ── Audio tower config ─────────────────────────────────────────────

AUDIO_HF_ID = 'facebook/wav2vec2-base'
# Wav2Vec2-base output is (B, T_out, 768) where T_out = floor(samples / 320).
# At 16 kHz that's a 50 Hz step → 20 ms per token. Lahner clips are 3s,
# so each clip yields ~150 audio tokens before chunking; AudioWrapper's
# auto-chunking is harmless here (max_duration_sec defaults to 60s).
AUDIO_LAYER = 'encoder.layers.6'  # mid-encoder block; speech features


# Region → layer mapping spans both towers. The region_modality_map
# below tells BrainScoreModel which tower owns each layer for cross-
# tower routing in multi-modality scoring.
REGION_LAYER_MAP = {
    # Vision tower (V-JEPA v1 ViT-L; mid-encoder block 16 was the
    # per-voxel layer-sweep peak on Lahner2024 — see CLAUDE.md notes).
    'V1': 'backbone.blocks.4',
    'V4': 'backbone.blocks.15',
    'IT': 'backbone.blocks.16',
    # Audio tower (Wav2Vec2-base; mid encoder block).
    'A1': AUDIO_LAYER,
}

REGION_MODALITY_MAP = {
    'V1': 'video',
    'V4': 'video',
    'IT': 'video',
    'A1': 'audio',
}


def _build_video_wrapper(identifier: str):
    """Reuse the V-JEPA v1 backbone + preprocessing from the standalone
    registration. Identifier passed in so VideoWrapper's cache key is
    distinct from the standalone vjepa1-vitl entry — we don't share
    backbone_id because the benchmark routing differs."""
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    from brainscore.models.vjepa_v1.model import (
        _BCTPermuteAdapter, _default_cache_dir, _download_checkpoint,
        _load_vjepa_v1_vitl, _make_preprocessing, _vjepa_v1_post_hook,
        CHECKPOINT_URL, NUM_INPUT_FRAMES,
    )

    cache = _default_cache_dir()
    ckpt = _download_checkpoint(CHECKPOINT_URL, cache / 'vitl16.pth.tar')
    backbone = _load_vjepa_v1_vitl(ckpt).eval()
    model = _BCTPermuteAdapter(backbone).eval()
    preprocessing = _make_preprocessing()
    wrapper = VideoWrapper(
        model=model,
        preprocessing=preprocessing,
        identifier=f'{identifier}-video',
        # Share the V-JEPA v1 standalone backbone cache key — same weights,
        # same forward pass; activations cached under one identifier.
        backbone_id='vjepa1-vitl',
        num_frames=NUM_INPUT_FRAMES,
        post_hook_fn=_vjepa_v1_post_hook,
        batch_size=4,
    )
    return wrapper, model


def _build_audio_wrapper(identifier: str):
    """Wrap Wav2Vec2-base with AudioWrapper. AudioWrapper handles the
    whole audio pipeline (load WAV, resample, processor, hook, batch,
    cache, package) — symmetric with the other wrappers."""
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    from scipy.io import wavfile

    model = Wav2Vec2Model.from_pretrained(AUDIO_HF_ID).eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_HF_ID)

    def _wav_loader(audio_path: str, target_rate: int) -> np.ndarray:
        sr, wav = wavfile.read(audio_path)
        if sr != target_rate:
            raise ValueError(
                f"WAV at {audio_path} has sr={sr}, expected {target_rate}. "
                f"Re-extract via prepare_audio_tracks.py with --target-rate "
                f"{target_rate}."
            )
        # int16 → float32 in [-1, 1]
        if wav.dtype.kind == 'i':
            max_val = np.iinfo(wav.dtype).max
            wav = wav.astype(np.float32) / max_val
        else:
            wav = wav.astype(np.float32)
        # Wav2Vec2 expects mono
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        return wav

    wrapper = AudioWrapper(
        model=model,
        processor=processor,
        identifier=f'{identifier}-audio',
        backbone_id='wav2vec2-base',
        layer_aggregation='time_series',  # keep T axis for temporal alignment
        max_duration_sec=60.0,
        batch_size=4,
        audio_loader=_wav_loader,
        audio_input_key='input_values',
    )
    return wrapper, model


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'vjepa1-wav2vec2'

    video_wrapper, video_model = _build_video_wrapper(identifier)
    audio_wrapper, audio_model = _build_audio_wrapper(identifier)

    return BrainScoreModel(
        identifier=identifier,
        # No single fused model — preprocessors hold their own backbones.
        # `model` is unused when every preprocessor is an extractor
        # (PytorchWrapper / TextWrapper / VideoWrapper / AudioWrapper),
        # which all duck-type as having an ``identifier`` attribute.
        model=None,
        region_layer_map=REGION_LAYER_MAP,
        region_modality_map=REGION_MODALITY_MAP,
        preprocessors={
            'video': video_wrapper,
            'audio': audio_wrapper,
        },
        # Both towers are required — the multimodal score wouldn't make
        # sense if a benchmark provided only one stream.
        required_modalities={'video', 'audio'},
        activations_model=None,
        visual_degrees=8,
    )
