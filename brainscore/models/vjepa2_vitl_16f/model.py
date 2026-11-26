"""V-JEPA 2 ViT-L, a speed variant of ``vjepa2-vitl``: the same checkpoint (facebook/vjepa2-vitl-fpc64-256), the
same region map and the same ``cortex`` blocks, with 16 input frames and a bf16 forward pass.

Limit: a speed variant of vjepa2-vitl, not the same subject: 16 frames spread evenly over each clip (vjepa2-vitl:
64), so 8 temporal steps instead of 32; its values differ from vjepa2-vitl's.
Limit: the forward pass runs under bf16 autocast on CUDA (vjepa2-vitl: float32); the recorded block outputs are
converted to float32.
On the Mario clips (89 frames, 1.49 s) the 64-frame form needs about 80 GPU hours for the five participants on one
A10G; this form about 15 times less.
"""
import functools
from typing import List

import numpy as np
import torch

from brainscore import BrainScoreModel
from brainscore.models._shared import fit_frames, tubelet_spatial_mean
from brainscore.models.vjepa.model import NUM_SPATIAL_PATCHES, region_layers

NUM_INPUT_FRAMES = 16
NUM_TEMPORAL_STEPS = NUM_INPUT_FRAMES // 2      # tubelet size 2
CHECKPOINT = 'facebook/vjepa2-vitl-fpc64-256'


def _make_preprocessing(processor):
    def preprocess(frames: List[np.ndarray]):
        out = processor(fit_frames(frames, NUM_INPUT_FRAMES), return_tensors='pt')
        key = 'pixel_values_videos' if 'pixel_values_videos' in out else 'pixel_values'
        return out[key][0]  # (T, C, H, W)
    return preprocess


_post_hook = functools.partial(tubelet_spatial_mean, temporal_steps=NUM_TEMPORAL_STEPS, spatial_patches=NUM_SPATIAL_PATCHES)


def bf16_forward(model):
    """The model's forward under bf16 autocast on CUDA (float32 elsewhere); module names are unchanged."""
    forward = model.forward

    def autocast_forward(*args, **kwargs):
        device = next(model.parameters()).device.type
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            return forward(*args, **kwargs)
    model.forward = autocast_forward
    return model


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'vjepa2-vitl-16f'
    from brainscore.models._downloads import hf_preflight
    from transformers import VJEPA2Model, VJEPA2VideoProcessor
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    download = hf_preflight(identifier, CHECKPOINT, 1.3)
    vjepa = VJEPA2Model.from_pretrained(CHECKPOINT, torch_dtype=torch.float32, **download)
    processor = VJEPA2VideoProcessor.from_pretrained(CHECKPOINT, **download)
    video_wrapper = VideoWrapper(model=vjepa, preprocessing=_make_preprocessing(processor), identifier=identifier,
                                 num_frames=NUM_INPUT_FRAMES, post_hook_fn=_post_hook,
                                 forward_kwargs={'skip_predictor': True})
    bf16_forward(vjepa)       # after the wrapper placed the model on its device
    return BrainScoreModel(identifier=identifier, model=vjepa, region_layer_map=region_layers(),
                           extractor={'video': video_wrapper})
