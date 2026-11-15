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

import functools
from typing import List

import numpy as np
import torch

from brainscore import BrainScoreModel
from brainscore.models._shared import fit_frames, tubelet_spatial_mean


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

    Frames are fitted to exactly NUM_INPUT_FRAMES (``fit_frames``), so every
    video yields the same 32 temporal tubelets regardless of source FPS.
    """
    def preprocess(frames: List[np.ndarray]):
        out = processor(fit_frames(frames, NUM_INPUT_FRAMES), return_tensors='pt')
        key = 'pixel_values_videos' if 'pixel_values_videos' in out else 'pixel_values'
        return out[key][0]  # (T, C, H, W)
    return preprocess


# V-JEPA2 tokens are all spatial patches at time 0, then time 1, ... (as VideoMAE).
_vjepa_post_hook = functools.partial(tubelet_spatial_mean, temporal_steps=NUM_TEMPORAL_STEPS,
                                     spatial_patches=NUM_SPATIAL_PATCHES)


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'vjepa2-vitl'
    from brainscore.models._downloads import hf_preflight
    checkpoint = 'facebook/vjepa2-vitl-fpc64-256'
    download = hf_preflight(identifier, checkpoint, 1.3)

    from transformers import VJEPA2Model, VJEPA2VideoProcessor
    from brainscore.model_helpers.video_wrapper import VideoWrapper

    vjepa = VJEPA2Model.from_pretrained(
        checkpoint,
        torch_dtype=torch.float32,
        **download,
    )
    processor = VJEPA2VideoProcessor.from_pretrained(checkpoint, **download)

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
    )

    return BrainScoreModel(
        identifier=identifier,
        model=vjepa,
        region_layer_map=REGION_LAYER_MAP,
        extractor={
            'video': video_wrapper,
        },
    )
