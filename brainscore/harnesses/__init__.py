"""Environments, policies, a human subject and adapters for external libraries.

Device specifics stay out of ``core``; they live here. ``robotics`` holds the
DROID-shaped embodied rows and the helpers that pack and validate them;
``grid_game`` and ``gymnasium_harness`` are environments with a closed-loop
driver; ``multi_agent`` passes outputs between subjects; ``droid_policy`` and
``libero_policy`` are ready-made policies; ``human`` and ``human_replay`` are a
prototype human subject; ``adapter``, ``coggym`` and ``ewok`` adapt external
experiment libraries to the common channels.
"""
from . import robotics
from . import human
from . import grid_game
from . import multi_agent
