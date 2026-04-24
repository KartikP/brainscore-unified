"""
VideoMAE base registered as a BrainScoreModel — first native-temporal
video model in the unified interface.

VideoMAE (Tong et al., NeurIPS 2022) is a masked-autoencoder pretrained
on Kinetics-400 videos. It takes a ``(B, T=16, C=3, H=224, W=224)``
tensor, tubelet-embeds (tubelet_size=2 → 8 temporal steps × 14×14 spatial
patches = 8 × 196 = 1568 tokens per video), and processes the result
through a 12-layer transformer encoder.

This is the first unified-interface model that uses ``VideoWrapper``
instead of ``PytorchWrapper`` / ``VLMVisionWrapper`` / ``TextWrapper``.
Scoring on Lahner2024 will reveal whether native temporal processing
meaningfully beats the frame-aggregation path (CLIP 0.4606 visual-ROI).

Architecture details:
    - Input: (B, 16, 3, 224, 224) video tensor, normalized per VideoMAE
      processor
    - Encoder: 12 transformer layers, hidden_size=768, 12 heads
    - Hook output at encoder.layer.{N}: (B, 1568, 768) — temporal and
      spatial tokens interleaved (8 temporal × 196 spatial)
    - Post-hook reshape: (B, 1568, 768) → (B, 8, 196, 768) → mean over
      spatial → (B, 8, 768) per-time-step features

Region-to-layer mapping follows the CLIP convention — map brain regions
to encoder layers spread through the network depth. These are placeholder
choices; Layer Mapping Explorer would select empirically.
"""

from typing import List

import numpy as np
import torch

from brainscore_core.model_interface import BrainScoreModel


REGION_LAYER_MAP = {
    'V1': 'encoder.layer.2',
    'V2': 'encoder.layer.4',
    'V4': 'encoder.layer.7',
    'IT': 'encoder.layer.11',
}

# VideoMAE base config: 16 input frames, tubelet_size=2 → 8 temporal steps,
# 224×224 image with patch_size=16 → 14×14 = 196 spatial patches per tubelet
NUM_INPUT_FRAMES = 16
NUM_TEMPORAL_STEPS = NUM_INPUT_FRAMES // 2  # tubelet_size=2
NUM_SPATIAL_PATCHES = (224 // 16) ** 2  # 14*14 = 196


def _make_preprocessing(processor):
    """Return a preprocessing callable compatible with VideoWrapper.

    Takes a list of HxWx3 RGB numpy frames, returns a (T, C, H, W) tensor
    ready for VideoMAE. The processor handles resizing to 224x224 and
    normalization with the VideoMAE-specific mean/std.
    """
    def preprocess(frames: List[np.ndarray]):
        # processor.__call__ expects a list of frames for one video OR a
        # list-of-lists for a batch. We pass a single-video list and get
        # back a pixel_values of shape (1, T, C, H, W); drop batch dim.
        out = processor(list(frames), return_tensors='pt')
        return out['pixel_values'][0]  # (T, C, H, W)
    return preprocess


def _videomae_post_hook(arr: np.ndarray) -> np.ndarray:
    """Reshape VideoMAE hook output from interleaved (B, T×S, H) to (B, T, H).

    VideoMAE's encoder outputs tokens ordered as all spatial patches at
    time 0, then all spatial patches at time 1, etc. Reshape to
    (B, T, S, H) and mean-pool across the spatial dimension to get one
    feature vector per temporal step.
    """
    B, seq_len, hidden = arr.shape
    if seq_len != NUM_TEMPORAL_STEPS * NUM_SPATIAL_PATCHES:
        # Defensive: if VideoMAE config ever differs, derive T and S
        # from the sequence length (assuming 196 spatial per tubelet).
        if seq_len % NUM_SPATIAL_PATCHES == 0:
            t_steps = seq_len // NUM_SPATIAL_PATCHES
            s_steps = NUM_SPATIAL_PATCHES
        elif seq_len % NUM_TEMPORAL_STEPS == 0:
            t_steps = NUM_TEMPORAL_STEPS
            s_steps = seq_len // NUM_TEMPORAL_STEPS
        else:
            raise ValueError(
                f"VideoMAE post-hook: unable to infer (T, S) from "
                f"seq_len={seq_len}.")
    else:
        t_steps = NUM_TEMPORAL_STEPS
        s_steps = NUM_SPATIAL_PATCHES

    reshaped = arr.reshape(B, t_steps, s_steps, hidden)
    return reshaped.mean(axis=2)  # mean over spatial patches → (B, T, H)


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'videomae-base'

    from transformers import VideoMAEModel, VideoMAEImageProcessor
    from brainscore.model_helpers.video_wrapper import VideoWrapper

    videomae = VideoMAEModel.from_pretrained(
        'MCG-NJU/videomae-base',
        torch_dtype=torch.float32,  # FP32 is fine for 86M params
    )
    processor = VideoMAEImageProcessor.from_pretrained(
        'MCG-NJU/videomae-base')

    preprocessing = _make_preprocessing(processor)

    video_wrapper = VideoWrapper(
        model=videomae,
        preprocessing=preprocessing,
        identifier=identifier,
        num_frames=NUM_INPUT_FRAMES,
        post_hook_fn=_videomae_post_hook,
        batch_size=4,  # 4 videos per forward — fits on A10G 24GB
    )

    return BrainScoreModel(
        identifier=identifier,
        model=videomae,
        region_layer_map=REGION_LAYER_MAP,
        preprocessors={
            # 'video' modality routes a stimulus set with ``video_path``
            # column to this VideoWrapper (see BrainScoreModel.COLUMN_TO_MODALITY).
            # The wrapper handles frame sampling + forward pass + hook + packaging.
            'video': video_wrapper,
        },
        required_modalities={'video'},
        activations_model=None,  # VideoWrapper IS the activations_model
        visual_degrees=8,
    )
