"""Robotics environment harness: DROID-shaped rows.

``core`` knows no device schema. A robot step is a row: cameras are named streams
on ``vision:image`` (``NamedStreams({'wrist': rgb, 'exterior_1': rgb})``), body
state is a mapping on ``proprioception`` (``joint_position``, ``cartesian_position``,
``gripper_position``, optional velocities and ``base_*`` for mobile robots), the
task text is on ``instruction`` and the reward on ``reward``. :func:`robotics_row`
packs and validates one.
"""
from typing import Any, Dict, Mapping, Optional

from brainscore.robotics import validate_cameras
from brainscore_core.streaming import NamedStreams

__all__ = [
    "validate_cameras",
    "robotics_row",
]


def robotics_row(
    cameras: Mapping[str, Any], proprioception: Mapping[str, Any],
    instruction: Optional[str] = None, reward: Optional[float] = None,
) -> Dict[str, Any]:
    """Pack named camera frames and body state into one closed-loop row.

    Cameras are validated. ``instruction`` and ``reward`` are added when given.
    """
    validate_cameras(cameras)
    row: Dict[str, Any] = {"vision:image": NamedStreams(cameras),
                           "proprioception": dict(proprioception)}
    if instruction is not None:
        row["instruction"] = instruction
    if reward is not None:
        row["reward"] = reward
    return row


# Public robotics integration surface, alongside the legacy envelope helpers.
from brainscore.robotics import (ActionSpec, droid_steps, evaluate_droid_episode, staged_droid_episodes)
__all__ += ["ActionSpec", "droid_steps", "evaluate_droid_episode", "staged_droid_episodes"]
