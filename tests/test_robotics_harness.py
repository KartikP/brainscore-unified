"""Tests for the robotics environment harness: DROID-shaped rows.

``core`` knows no device schema; a robot step is a row of named camera streams,
a proprioception mapping, an instruction and a reward.
"""
import numpy as np

from brainscore.harnesses.robotics import robotics_row
from brainscore_core.streaming import NamedStreams, events_row, row_events


def _cams():
    return {"wrist": np.zeros((180, 320, 3), dtype=np.uint8)}


def _proprio():
    return {"joint_position": np.zeros(7), "cartesian_position": np.zeros(6),
            "gripper_position": np.zeros(1)}


class TestRoboticsHarness:
    def test_row_packs_named_cameras_proprioception_instruction_and_reward(self):
        row = robotics_row(_cams(), _proprio(), instruction="pick up", reward=0.5)
        assert isinstance(row["vision:image"], NamedStreams) and "wrist" in row["vision:image"]
        assert set(row["proprioception"]) == {"joint_position", "cartesian_position", "gripper_position"}
        assert row["instruction"] == "pick up" and row["reward"] == 0.5

    def test_optional_entries_are_absent_when_not_given(self):
        assert set(robotics_row(_cams(), _proprio())) == {"vision:image", "proprioception"}

    def test_row_round_trips_through_events(self):
        row = robotics_row(_cams(), _proprio(), instruction="pick up")
        back = events_row(row_events(row, 10.0))
        assert set(back) == set(row) and set(back["vision:image"]) == {"wrist"}
