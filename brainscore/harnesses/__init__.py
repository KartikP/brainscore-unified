"""Environment and subject harnesses for the v1.5 interface.

A harness adapts a concrete subject or environment to the device-agnostic core
contract. v1.5 keeps all device specifics out of ``core``; they live here.

The robotics harness is the going-forward home for the DROID-shaped embodied
types and the helpers that pack them into ``EnvironmentStep.observation``. The
grid_game harness is a self-contained video-game environment with a closed-loop
driver. A human harness and other environment harnesses (Atari, browser) are
reserved.
"""
from . import robotics
from . import human
from . import grid_game
