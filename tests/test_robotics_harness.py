"""Tests for the v1.5 robotics environment harness.

The harness is the going-forward home for the DROID-shaped types and the helpers
that pack robotics inputs into the device-agnostic ``EnvironmentStep.observation``.
"""
import numpy as np

import brainscore_core.model_interface as core_mi
from brainscore_core.model_interface import EnvironmentResponse, EnvironmentStep
from brainscore.harnesses.robotics import (
    CameraFrame,
    Proprioception,
    build_environment_step,
    cameras_from_observation,
    droid_action,
    proprioception_from_observation,
    robotics_observation,
)


def _cams():
    return {"wrist": CameraFrame(rgb=np.zeros((180, 320, 3), dtype=np.uint8))}


def _proprio():
    return Proprioception(
        joint_position=np.zeros(7),
        cartesian_position=np.zeros(6),
        gripper_position=np.zeros(1),
    )


class TestRoboticsHarness:
    def test_types_reexported_from_harness(self):
        # New code imports the DROID types from the harness. During the
        # deprecation window they are the same objects as core's.
        assert CameraFrame is core_mi.CameraFrame
        assert Proprioception is core_mi.Proprioception

    def test_robotics_observation_packs_payload(self):
        obs = robotics_observation(_cams(), _proprio())
        assert "cameras" in obs and "proprioception" in obs

    def test_build_environment_step_uses_device_agnostic_envelope(self):
        step = build_environment_step(
            _cams(), _proprio(), instruction="pick up", step_num=2, is_first=True
        )
        assert isinstance(step, EnvironmentStep)
        # The robotics specifics are in the observation payload; the harness used
        # the generic envelope rather than the deprecated core fields.
        assert step.observation is not None and "cameras" in step.observation
        assert step.cameras is None and step.proprioception is None
        assert step.instruction == "pick up" and step.step_num == 2

    def test_read_back_helpers(self):
        step = build_environment_step(_cams(), _proprio())
        assert "wrist" in cameras_from_observation(step.observation)
        assert proprioception_from_observation(step.observation) is not None

    def test_droid_action_builds_response(self):
        resp = droid_action(np.zeros(7), metadata={"v": 1})
        assert isinstance(resp, EnvironmentResponse)
        assert resp.action.shape == (7,)
        assert resp.metadata["v"] == 1
