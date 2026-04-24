"""
V-JEPA2 ViT-L registered as a BrainScoreModel — second native-temporal
video model in the unified interface, and the first non-pixel-reconstruction
video backbone.

V-JEPA2 (Bardes et al., 2025) is a joint-embedding predictive architecture
pretrained to predict target representations (not pixels) of masked
space-time regions given visible context. Unlike VideoMAE, which
reconstructs raw pixels and is known to transfer poorly to neural
prediction tasks, V-JEPA operates in representation space — the same
family of objectives the EPFL group (10.1101/2025.07.22.664908v3) reports
hits ~0.51 normalized correlation on BOLDMoments, beating CLIP and
VideoMAE.

Architecture details:
    - Checkpoint: facebook/vjepa2-vitl-fpc64-256 (ViT-L)
    - Input: (B, 64, 3, 256, 256) video, normalized per VideoProcessor
    - Encoder: 24 transformer layers, hidden_size=1024, 16 heads
    - Tubelet size 2 → 32 temporal steps
    - Patch size 16 at 256px → 16*16 = 256 spatial patches per tubelet
    - Hook output at encoder.layer.{N}: (B, 32*256=8192, 1024), wrapped as
      tuple (VideoWrapper unwraps); post-hook reshape to (B, 32, 256, 1024)
      and spatial-mean to (B, 32, 1024)
    - Forward is called with ``skip_predictor=True`` to skip the 12-layer
      predictor head that isn't used for feature extraction

Region-to-layer mapping follows the CLIP/VideoMAE convention — map brain
regions to encoder layers spread through the network depth. These are
placeholder choices; the Layer Mapping Explorer would select empirically.
Lahner2024 only exercises the IT mapping today.
"""

from typing import List

import numpy as np
import torch

from brainscore_core.model_interface import BrainScoreModel


REGION_LAYER_MAP = {
    'V1': 'encoder.layer.4',
    'V2': 'encoder.layer.9',
    'V4': 'encoder.layer.15',
    # IT → encoder.layer.16 by analogy with V-JEPA v1 per-voxel sweep on
    # Lahner2024 (layer 16 peaks at 0.61 raw ROI r vs last layer 0.54 —
    # see 2026-04-24 replication note). V-JEPA v2 has the same 24-layer
    # ViT-L architecture; a dedicated v2 sweep would be a small follow-up.
    'IT': 'encoder.layer.16',
}

# V-JEPA2-L config: 64 input frames, tubelet_size=2 → 32 temporal steps,
# 256×256 image with patch_size=16 → 16×16 = 256 spatial patches per tubelet.
NUM_INPUT_FRAMES = 64
NUM_TEMPORAL_STEPS = NUM_INPUT_FRAMES // 2  # tubelet_size=2
NUM_SPATIAL_PATCHES = (256 // 16) ** 2  # 16*16 = 256


def _make_preprocessing(processor):
    """Return a preprocessing callable compatible with VideoWrapper.

    Takes a list of HxWx3 RGB numpy frames, returns a (T, C, H, W) tensor
    ready for V-JEPA2. The processor resizes to 256x256 and normalizes
    with the V-JEPA-specific mean/std.

    VideoWrapper's default frame sampler returns fewer than ``num_frames``
    when the underlying video has fewer frames (e.g., Lahner2024 has some
    clips at 15 FPS × 3 s = 45 frames). V-JEPA2 tubelets would then vary
    in count per video, breaking the cross-video stack in ``_package``.
    We pad/truncate to exactly NUM_INPUT_FRAMES here so every video yields
    the same 32 temporal tubelets regardless of source FPS.
    """
    def preprocess(frames: List[np.ndarray]):
        frames = list(frames)
        if len(frames) == 0:
            raise ValueError("V-JEPA preprocess received empty frame list")
        if len(frames) < NUM_INPUT_FRAMES:
            pad_n = NUM_INPUT_FRAMES - len(frames)
            frames = frames + [frames[-1]] * pad_n  # repeat last frame
        elif len(frames) > NUM_INPUT_FRAMES:
            # Evenly subsample to NUM_INPUT_FRAMES — guards against the
            # target-fps sampling path returning more than we expect.
            idxs = [int(round(i * (len(frames) - 1) / (NUM_INPUT_FRAMES - 1)))
                    for i in range(NUM_INPUT_FRAMES)]
            frames = [frames[i] for i in idxs]
        out = processor(frames, return_tensors='pt')
        key = 'pixel_values_videos' if 'pixel_values_videos' in out else 'pixel_values'
        return out[key][0]  # (T, C, H, W)
    return preprocess


def _vjepa_post_hook(arr: np.ndarray) -> np.ndarray:
    """Reshape V-JEPA2 hook output from interleaved (B, T*S, H) to (B, T, H).

    V-JEPA2 tokens are ordered as all spatial patches at time 0, then all
    spatial patches at time 1, etc. (same convention as VideoMAE). Reshape
    to (B, T, S, H) and mean-pool across the spatial dimension.
    """
    B, seq_len, hidden = arr.shape
    if seq_len == NUM_TEMPORAL_STEPS * NUM_SPATIAL_PATCHES:
        t_steps = NUM_TEMPORAL_STEPS
        s_steps = NUM_SPATIAL_PATCHES
    elif seq_len % NUM_SPATIAL_PATCHES == 0:
        t_steps = seq_len // NUM_SPATIAL_PATCHES
        s_steps = NUM_SPATIAL_PATCHES
    elif seq_len % NUM_TEMPORAL_STEPS == 0:
        t_steps = NUM_TEMPORAL_STEPS
        s_steps = seq_len // NUM_TEMPORAL_STEPS
    else:
        raise ValueError(
            f"V-JEPA post-hook: unable to infer (T, S) from seq_len={seq_len}.")

    reshaped = arr.reshape(B, t_steps, s_steps, hidden)
    return reshaped.mean(axis=2)  # mean over spatial patches → (B, T, H)


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'vjepa2-vitl'

    from transformers import VJEPA2Model, VJEPA2VideoProcessor
    from brainscore.model_helpers.video_wrapper import VideoWrapper

    checkpoint = 'facebook/vjepa2-vitl-fpc64-256'
    vjepa = VJEPA2Model.from_pretrained(
        checkpoint,
        torch_dtype=torch.float32,
    )
    processor = VJEPA2VideoProcessor.from_pretrained(checkpoint)

    preprocessing = _make_preprocessing(processor)

    video_wrapper = VideoWrapper(
        model=vjepa,
        preprocessing=preprocessing,
        identifier=identifier,
        num_frames=NUM_INPUT_FRAMES,
        post_hook_fn=_vjepa_post_hook,
        # skip_predictor avoids running the 12-layer predictor head at
        # inference time; it's only needed during self-supervised pretraining.
        forward_kwargs={'skip_predictor': True},
        # 8192 tokens × 1024 hidden × 24 layers is attention-heavy; one
        # video per forward keeps us comfortably under A10G 24GB.
        batch_size=1,
    )

    return BrainScoreModel(
        identifier=identifier,
        model=vjepa,
        region_layer_map=REGION_LAYER_MAP,
        preprocessors={
            # 'video' modality routes a stimulus set with ``video_path``
            # column to this VideoWrapper (see BrainScoreModel.COLUMN_TO_MODALITY).
            'video': video_wrapper,
        },
        required_modalities={'video'},
        activations_model=None,  # VideoWrapper IS the activations_model
        visual_degrees=8,
    )
