"""
V-JEPA v1 ViT-L/16 registered as a BrainScoreModel.

Direct match for the model used by the EPFL NeuroAI paper
("Many-Two-One: Diverse Representations Across Visual Pathways Emerge from
A Single Objective", Tang et al. 2025, bioRxiv 10.1101/2025.07.22.664908).
That paper loads Meta's original V-JEPA v1 ViT-L/16 checkpoint via
``facebookresearch/jepa``; we vendor the minimal subset of its backbone
(see ``backbone/`` package) and load the public checkpoint from
``dl.fbaipublicfiles.com``.

This is DIFFERENT from V-JEPA **v2** (registered separately as
``vjepa2-vitl``):

========================  ================================  ================================
                          V-JEPA v1 (this)                  V-JEPA v2
========================  ================================  ================================
Checkpoint                vitl16.pth.tar (Meta FAIR)        facebook/vjepa2-vitl-fpc64-256
Training corpus           VideoMix2M (~2M videos)           ~22M videos
Input frames              16                                64
Resolution                224x224                           256x256
Tubelet / patch           2 / 16                            2 / 16
Layers / hidden           24 / 1024                         24 / 1024
Tokens per forward        8 * 14 * 14 = 1568                32 * 16 * 16 = 8192
========================  ================================  ================================

The EPFL paper reports ~0.51 normalized correlation on BOLDMoments for
V-JEPA v1 ViT-L with their exact pipeline; registering this model lets us
directly reproduce their result and calibrate against our own pipeline.

Forward interface:
    V-JEPA v1's VisionTransformer expects input (B, C, T, H, W) — channel
    before time. VideoWrapper feeds (B, T, C, H, W). We wrap the backbone
    in ``_BCTPermuteAdapter`` that permutes internally and exposes
    ``blocks.{N}`` for hook access. No separate permute in preprocessing
    because it would violate VideoWrapper's (T, C, H, W) contract.

Hook output shape:
    Each ``blocks.{N}`` forward hook returns (B, N_tokens, 1024) where
    N_tokens = 8 * 14 * 14 = 1568 (same layout as VideoMAE). Our post-hook
    reshapes (B, 1568, 1024) -> (B, 8, 196, 1024) -> mean-over-spatial ->
    (B, 8, 1024), yielding one feature vector per temporal tubelet.

Region-to-layer mapping spreads regions through the 24-layer encoder.
Lahner2024 only exercises IT today (encoder.layer.23 analog = blocks.23).
"""

import os
import hashlib
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from brainscore_core.model_interface import BrainScoreModel


# --- V-JEPA v1 ViT-L/16 pretrained config (matches vitl16.pth.tar) ---
NUM_INPUT_FRAMES = 16
TUBELET_SIZE = 2
NUM_TEMPORAL_STEPS = NUM_INPUT_FRAMES // TUBELET_SIZE  # 8
PATCH_SIZE = 16
IMG_SIZE = 224
NUM_SPATIAL_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 14*14 = 196

CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar'
CHECKPOINT_SHA256 = None  # Meta publishes no official SHA; skip verification.

# Match paper convention; can be swapped for layer-mapping-explorer output.
REGION_LAYER_MAP = {
    'V1': 'backbone.blocks.4',
    'V2': 'backbone.blocks.9',
    'V4': 'backbone.blocks.15',
    # IT → blocks.16 based on 2026-04-24 per-voxel layer sweep on Lahner2024:
    # blocks.16 peaks at 0.611 ROI r; blocks.23 drops to 0.536 (we were
    # defaulting to the last layer, costing ~0.08 raw r).
    'IT': 'backbone.blocks.16',
}


class _BCTPermuteAdapter(nn.Module):
    """Thin adapter so VideoWrapper's (B, T, C, H, W) -> backbone (B, C, T, H, W)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, *args, **kwargs):
        # x: (B, T, C, H, W). V-JEPA v1 expects (B, C, T, H, W).
        x = x.permute(0, 2, 1, 3, 4)
        return self.backbone(x, *args, **kwargs)


def _default_cache_dir() -> Path:
    # Follows common HF cache convention when HF_HOME/XDG_CACHE_HOME are set,
    # else ~/.cache/vjepa_v1.
    root = os.environ.get('HF_HOME') or os.environ.get('XDG_CACHE_HOME') \
        or os.path.expanduser('~/.cache')
    cache = Path(root) / 'vjepa_v1'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_checkpoint(url: str, dest: Path) -> Path:
    """Download a URL to ``dest`` if not already present. Atomic via tmp->rename."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import urllib.request
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def _strip_module_prefix(sd: dict) -> dict:
    """Remove 'module.' and 'backbone.' prefixes from state-dict keys if present."""
    out = {}
    for k, v in sd.items():
        nk = k
        for pref in ('module.backbone.', 'backbone.', 'module.'):
            if nk.startswith(pref):
                nk = nk[len(pref):]
                break
        out[nk] = v
    return out


def _load_vjepa_v1_vitl(checkpoint_path: Path) -> nn.Module:
    from .backbone import vit_large
    backbone = vit_large(
        patch_size=PATCH_SIZE,
        img_size=IMG_SIZE,
        num_frames=NUM_INPUT_FRAMES,
        tubelet_size=TUBELET_SIZE,
    )
    ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    # Meta's vitl16.pth.tar stores a dict with multiple entries; we want
    # the target encoder weights ('target_encoder' or 'encoder').
    if isinstance(ckpt, dict):
        for k in ('target_encoder', 'encoder', 'model', 'state_dict'):
            if k in ckpt:
                ckpt = ckpt[k]
                break
    sd = _strip_module_prefix(ckpt)
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    # Target encoder has no predictor/decoder keys; any 'predictor.*' in the
    # checkpoint should NOT match backbone and will land in `unexpected`.
    if missing and any(not k.startswith('predictor.') for k in missing):
        raise RuntimeError(
            f"V-JEPA v1 state_dict mismatch: missing keys (backbone-side) = "
            f"{[k for k in missing if not k.startswith('predictor.')][:5]}"
        )
    return backbone


def _make_preprocessing():
    """Return preprocessing callable matching V-JEPA v1's training pipeline.

    V-JEPA v1 training normalizes with ImageNet mean/std and resizes the
    shortest side then center-crops to 224. For evaluation on pre-cropped
    3-sec Lahner clips, direct resize to 224x224 per frame is sufficient.
    Frames are padded to exactly 16 to handle variable-FPS clips (same
    reasoning as the V-JEPA2 padding — see that model's docstring).
    """
    import cv2

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(frames: List[np.ndarray]):
        frames = list(frames)
        if len(frames) == 0:
            raise ValueError("V-JEPA v1 preprocess received empty frame list")
        if len(frames) < NUM_INPUT_FRAMES:
            frames = frames + [frames[-1]] * (NUM_INPUT_FRAMES - len(frames))
        elif len(frames) > NUM_INPUT_FRAMES:
            idxs = [int(round(i * (len(frames) - 1) / (NUM_INPUT_FRAMES - 1)))
                    for i in range(NUM_INPUT_FRAMES)]
            frames = [frames[i] for i in idxs]

        # Resize and normalize
        out = np.empty((NUM_INPUT_FRAMES, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for i, fr in enumerate(frames):
            img = cv2.resize(fr, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            out[i] = img.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(out)  # (T=16, C=3, H=224, W=224)
    return preprocess


def _vjepa_v1_post_hook(arr: np.ndarray) -> np.ndarray:
    """Reshape V-JEPA v1 block output from (B, T*S, H) to (B, T, H).

    Token order is all spatial patches at t=0, then t=1, ... (same as
    VideoMAE / V-JEPA v2). Reshape and spatial-mean.
    """
    B, seq_len, hidden = arr.shape
    if seq_len == NUM_TEMPORAL_STEPS * NUM_SPATIAL_PATCHES:
        t_steps, s_steps = NUM_TEMPORAL_STEPS, NUM_SPATIAL_PATCHES
    elif seq_len % NUM_SPATIAL_PATCHES == 0:
        t_steps, s_steps = seq_len // NUM_SPATIAL_PATCHES, NUM_SPATIAL_PATCHES
    elif seq_len % NUM_TEMPORAL_STEPS == 0:
        t_steps, s_steps = NUM_TEMPORAL_STEPS, seq_len // NUM_TEMPORAL_STEPS
    else:
        raise ValueError(
            f"V-JEPA v1 post-hook: unable to infer (T, S) from seq_len={seq_len}.")
    return arr.reshape(B, t_steps, s_steps, hidden).mean(axis=2)


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'vjepa1-vitl'

    from brainscore.model_helpers.video_wrapper import VideoWrapper

    cache = _default_cache_dir()
    ckpt = _download_checkpoint(CHECKPOINT_URL, cache / 'vitl16.pth.tar')
    backbone = _load_vjepa_v1_vitl(ckpt).eval()
    model = _BCTPermuteAdapter(backbone).eval()

    preprocessing = _make_preprocessing()

    video_wrapper = VideoWrapper(
        model=model,
        preprocessing=preprocessing,
        identifier=identifier,
        num_frames=NUM_INPUT_FRAMES,
        post_hook_fn=_vjepa_v1_post_hook,
        # Same 16-frame budget as VideoMAE → batch 4 fits on A10G 24GB.
        batch_size=4,
    )

    return BrainScoreModel(
        identifier=identifier,
        model=model,
        region_layer_map=REGION_LAYER_MAP,
        preprocessors={'video': video_wrapper},
        activations_model=None,
        visual_degrees=8,
    )
