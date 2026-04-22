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
    'IT': 'encoder.layer.23',
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
    """
    def preprocess(frames: List[np.ndarray]):
        # VJEPA2VideoProcessor accepts a list of frames for one video and
        # returns pixel_values_videos of shape (1, T, C, H, W); drop batch dim.
        out = processor(list(frames), return_tensors='pt')
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
        activations_model=None,  # VideoWrapper IS the activations_model
        visual_degrees=8,
    )
