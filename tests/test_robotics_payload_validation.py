"""Camera payloads are validated by the robotics harness, not by core, so a
malformed frame is rejected with a clear error before it reaches a policy."""
import numpy as np
import pytest

from brainscore.harnesses.robotics import (
    CameraFrame, Proprioception, build_environment_step, validate_cameras)


def _body():
    return Proprioception(joint_position=np.zeros(7), cartesian_position=np.zeros(6),
                          gripper_position=np.zeros(1))


@pytest.mark.parametrize('rgb, match', [
    (np.zeros((2, 2), dtype='float64'), 'failed validation'),
    (np.zeros((8, 8, 4), dtype='uint8'), 'H, W, 3'),
    (np.zeros((3, 8, 8), dtype='uint8'), 'H, W, 3'),
])
def test_malformed_camera_payload_rejected(rgb, match):
    with pytest.raises(ValueError, match=match):
        build_environment_step({'wrist': CameraFrame(rgb=rgb)}, _body())


def test_valid_camera_payload_passes():
    step = build_environment_step(
        {'wrist': CameraFrame(rgb=np.zeros((8, 8, 3), dtype='uint8'))}, _body())
    assert 'wrist' in step.observation['cameras']


def test_cameras_must_be_a_mapping():
    with pytest.raises(ValueError, match='mapping'):
        validate_cameras([CameraFrame(rgb=np.zeros((8, 8, 3), dtype='uint8'))])
