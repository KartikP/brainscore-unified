"""The Super Mario Bros. agent of CNeuroMod's mario tutorials, a policy that plays the game.

``mario-ppo``: the PPO agent trained on 22 levels for 10 million steps
(``models/ppo_mario_all_levels_10000k_20251210_043953.pth`` of https://github.com/courtois-neuromod/mario.tutorials
at commit 9df41595, MIT licence, Copyright (c) 2025 CNeuromod); fetched from that commit and checked by its
sha256. ``mario-ppo-untrained``: the same network with its initial weights (orthogonal, ReLU gain, zero
biases; seed 0), a floor.

The network (``SimpleCNN`` of ``src/rl.py``): four 3 x 3 convolutions (32 channels, stride 2, ReLU), a linear
layer of 512 units (ReLU), an actor head (12 button sets, ``brainscore.harnesses.mario.ACTIONS``) and a
critic head. Its input: the last 4 frames, grey, 84 x 84 (area interpolation), values in [0, 1], as in its
training. A step of ``brainscore.harnesses.mario`` gives the 4 frames on ``vision:video``; the policy takes
the button set of the largest actor output, as the tutorial's evaluation does.

``cortex`` names conv2, conv3, conv4 and fc. During play a session that requests ``neural:cortex`` records the
layers in the policy's one forward pass, on the input the agent acted on. Passively the extractor takes a stack
file of the 4 grey frames (``.npy``) or a picture on ``vision:image``: one grey frame, repeated 4 times.
"""
import hashlib
from pathlib import Path

import numpy as np

from brainscore import BrainScoreModel

COMMIT = '9df415952ba2e3154941aa8768252799726d7843'
WEIGHTS = 'models/ppo_mario_all_levels_10000k_20251210_043953.pth'
SHA256 = 'd73d39e6bf7468c81ce303374742ed36d9d5247d8c321b52c4cee96f6516259b'
URL = f'https://raw.githubusercontent.com/courtois-neuromod/mario.tutorials/{COMMIT}/{WEIGHTS}'
CORTEX_LAYERS = ('conv2', 'conv3', 'conv4', 'fc')
STACK = 4


def build_network():
    import torch
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        def __init__(self, n_actions=12, input_channels=STACK):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            self.fc = nn.Linear(32 * 6 * 6, 512)
            self.actor = nn.Linear(512, n_actions)
            self.critic = nn.Linear(512, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
                x = self.relu(conv(x))
            x = self.relu(self.fc(x.flatten(1)))
            return self.actor(x)

    torch.manual_seed(0)
    network = SimpleCNN()
    for module in network.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
            nn.init.constant_(module.bias, 0)
    return network.eval()


def trained_weights() -> Path:
    """The checkpoint, fetched once from the pinned commit into the torch hub folder; refused on another sha256."""
    import torch
    path = Path(torch.hub.get_dir()) / 'mario-ppo' / f'{COMMIT[:8]}-{Path(WEIGHTS).name}'
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(URL, str(path), progress=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f'{path} has sha256 {digest}, not {SHA256} (the file at {URL}): delete it and load again.')
    return path


def grey(frame: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.resize(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY), (84, 84),
                      interpolation=cv2.INTER_AREA)


def load_inputs(paths) -> np.ndarray:
    """The network input of each file: a stack file (``.npy``, 4 x 84 x 84 uint8) or a picture (4 times)."""
    import cv2
    stacks = []
    for path in paths:
        if str(path).endswith('.npy'):
            stacks.append(np.load(path))
        else:
            picture = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            stacks.append(np.repeat(grey(picture)[None], STACK, axis=0))
    return np.stack(stacks).astype(np.float32) / 255.0


class Policy:
    """One step: the button set of the largest actor output for the last 4 frames. The one forward pass of the
    network is the pass a recording during play reads (``neural:cortex``)."""

    def __init__(self, network):
        self.network = network

    def __call__(self, row) -> dict:
        import torch
        frames = np.asarray(row['vision:video'])[-STACK:]
        stack = np.stack([grey(frame) for frame in frames])
        with torch.no_grad():
            logits = self.network(torch.from_numpy(stack[None].astype(np.float32) / 255.0))
        return {'motor': int(logits.argmax(dim=-1).item())}


def get_model(identifier: str) -> BrainScoreModel:
    import torch
    from brainscore.selection import CompositeSelector
    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
    network = build_network()
    if identifier == 'mario-ppo':
        network.load_state_dict(torch.load(trained_weights(), map_location='cpu', weights_only=True))
    elif identifier != 'mario-ppo-untrained':
        raise ValueError(f"unknown mario agent {identifier!r}; this plugin registers 'mario-ppo' and "
                         f"'mario-ppo-untrained'.")
    network.eval()
    # Limit: batch_size=1: PytorchWrapper pads a batch to its size, and a recording during a step keeps the padded rows.
    extractor = PytorchWrapper(model=network, preprocessing=load_inputs, identifier=identifier, device='cpu',
                               batch_size=1)
    return BrainScoreModel(
        identifier=identifier,
        model=network,
        region_layer_map={'cortex': CompositeSelector(tuple((layer, None) for layer in CORTEX_LAYERS))},
        extractor={'vision': extractor},
        policy=Policy(network),
        policy_in_channels=('vision:video',),
    )
