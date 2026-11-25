"""The untrained network of the CNeuroMod mario paper (Paugam et al. 2025, bioRxiv 10.1101/2025.11.28.691119).

The architecture and the weight initialization of their PPO agent (``code/src/models.py`` of
https://github.com/courtois-neuromod/mario_generalization, MIT licence, Copyright (c) 2025 CNeuromod):
four 3 x 3 convolutions (32 channels, stride 2, ReLU) and a linear layer of 512 units, on a stack of 4
grey 84 x 84 frames; orthogonal weights with the ReLU gain, zero biases. Untrained: its weights are drawn
once with seed 0.

Limit: the paper's untrained weights (``models/untrained_model.pt``) are not released; these are new draws
of the same initialization, so they are not the paper's weights.

Input: one clip on ``vision:video``. From the 16 consecutive frames at the middle of the clip, as the
paper's ``preprocess_frames`` (4 frames, every 4th, each the maximum of two consecutive frames; grey;
84 x 84; values in [0, 1]); the paper gives every frame of the play to the network, the benchmark one
stack per TR. ``cortex``: the layers the paper fits (conv2, conv3, conv4, linear).
"""
from pathlib import Path
from typing import List, Union

import numpy as np

from brainscore import BrainScoreModel

FRAMES = 16             # consecutive frames of one input, as the paper (60 frames per second)
DOWNSAMPLE = 4
STACK = FRAMES // DOWNSAMPLE
CORTEX_LAYERS = ('base.conv2', 'base.conv3', 'base.conv4', 'base.linear')


def build_network():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class BaseModel(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.conv1 = nn.Conv2d(n_in, 32, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.linear = nn.Linear(32 * 6 * 6, 512)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            return self.linear(x.view(x.size(0), -1))

    class Network(nn.Module):
        """PPO's network without its two output heads; a clip batch ``(B, 1, 4, 84, 84)`` goes in."""

        def __init__(self):
            super().__init__()
            self.base = BaseModel(STACK)
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                    nn.init.constant_(module.bias, 0)

        def forward(self, video):
            return self.base(video.flatten(0, 1))

    torch.manual_seed(0)
    return Network().eval()


def middle_frames(video_path: Union[str, Path], num_frames: int = FRAMES, target_fps=None) -> List[np.ndarray]:
    """``num_frames`` consecutive RGB frames at the middle of the clip; a shorter clip repeats its last frame."""
    import cv2
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f'cv2 could not open {video_path}')
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, (total - num_frames) // 2))
    frames = []
    while len(frames) < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise IOError(f'no frame in {video_path}')
    return frames + [frames[-1]] * (num_frames - len(frames))


def preprocess(frames: List[np.ndarray]):
    """The paper's ``preprocess_frames(frames, 4, 4)``: a ``(1, 4, 84, 84)`` tensor (one step, 4 channels)."""
    import cv2
    import torch
    grey = np.array([cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (84, 84)) for f in frames],
                    dtype=np.float32) / 255.0
    stack = [np.maximum(grey[-i * DOWNSAMPLE - 2], grey[-i * DOWNSAMPLE - 1]) for i in range(STACK - 1, -1, -1)]
    return torch.from_numpy(np.stack(stack))[None]


def one_step(output: np.ndarray) -> np.ndarray:
    """A layer's output ``(B, ...)`` as one time step ``(B, 1, units)``: one stack is one step."""
    return output.reshape(output.shape[0], 1, -1)


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'paugam2025-untrained-cnn'
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    from brainscore.selection import CompositeSelector
    network = build_network()
    extractor = VideoWrapper(model=network, preprocessing=preprocess, identifier=identifier, num_frames=FRAMES,
                             frame_sampler=middle_frames, post_hook_fn=one_step, batch_size=256)
    return BrainScoreModel(
        identifier=identifier,
        model=network,
        region_layer_map={'cortex': CompositeSelector(tuple((layer, None) for layer in CORTEX_LAYERS))},
        extractor={'video': extractor},
    )
