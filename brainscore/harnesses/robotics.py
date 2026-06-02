"""Robotics environment harness for the v1.5 device-agnostic interface.

v1.5 moved device schema out of ``core``: ``EnvironmentStep`` now carries a
generic, harness-defined ``observation`` payload rather than DROID-shaped
``cameras`` and ``proprioception`` fields. This module is the going-forward home
for the DROID-shaped robotics types and the helpers that pack them into, and
read them back out of, the generic envelope.

The DROID types are imported from ``core`` for one release (where they remain as
deprecated aliases) so existing code keeps working. New robotics code should
import them from here and build steps with :func:`build_environment_step`, which
uses the device-agnostic ``observation`` field rather than the deprecated core
fields.
"""
from typing import Any, Dict, Optional

# The DROID-shaped types live in core for one more release; re-export them here
# as their going-forward home so new robotics code imports them from the harness.
from brainscore_core.model_interface import (
    CameraFrame,
    EnvironmentResponse,
    EnvironmentStep,
    Proprioception,
)

__all__ = [
    "CameraFrame",
    "Proprioception",
    "robotics_observation",
    "build_environment_step",
    "droid_action",
    "cameras_from_observation",
    "proprioception_from_observation",
]


def robotics_observation(
    cameras: Dict[str, CameraFrame], proprioception: Proprioception
) -> Dict[str, Any]:
    """Pack DROID-shaped cameras and proprioception into a generic observation
    payload suitable for ``EnvironmentStep.observation``."""
    return {"cameras": cameras, "proprioception": proprioception}


def build_environment_step(
    cameras: Dict[str, CameraFrame],
    proprioception: Proprioception,
    instruction: Optional[str] = None,
    step_num: int = 0,
    is_first: bool = False,
    is_last: bool = False,
    is_terminal: bool = False,
    reward: Optional[float] = None,
    discount: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None,
) -> EnvironmentStep:
    """Construct a device-agnostic ``EnvironmentStep`` from DROID-shaped robotics
    inputs. The robotics specifics live here, in the harness; ``core`` sees only
    the generic ``observation`` payload, never the camera or joint layout."""
    return EnvironmentStep(
        observation=robotics_observation(cameras, proprioception),
        instruction=instruction,
        step_num=step_num,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        reward=reward,
        discount=discount,
        context=context or {},
    )


def droid_action(
    action, action_dict: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EnvironmentResponse:
    """Build an ``EnvironmentResponse`` carrying a DROID-compatible action."""
    return EnvironmentResponse(action=action, action_dict=action_dict, metadata=metadata)


def cameras_from_observation(observation) -> Optional[Dict[str, CameraFrame]]:
    """Read the cameras back out of a robotics observation payload, if present."""
    if isinstance(observation, dict):
        return observation.get("cameras")
    return None


def proprioception_from_observation(observation) -> Optional[Proprioception]:
    """Read the proprioception back out of a robotics observation payload."""
    if isinstance(observation, dict):
        return observation.get("proprioception")
    return None
