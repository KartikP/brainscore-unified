"""Closed-loop rows and ``process(row)`` dispatch on ``BrainScoreModel``.

A step is a row (a mapping of channel -> payload) following DROID
(https://droid-dataset.github.io/) for arm manipulation: named cameras on
``vision:image``, body state on ``proprioception``, the task on ``instruction``,
the reward on ``reward``. These tests cover policy dispatch, error handling when
no policy is registered, a multi-step rollout and the input-handler table.
"""

import numpy as np
import pytest

from brainscore import BrainScoreModel
from brainscore_core.streaming import NamedStreams


# -- Helpers ------------------------------------------------------------------

def _droid_row(instruction: str = "pick up the red block", reward=None):
    """A DROID-shaped row: the canonical 3-camera setup, 7-DOF Franka state and a
    language instruction."""
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    row = {
        'vision:image': NamedStreams({'exterior_1': frame, 'exterior_2': frame, 'wrist': frame}),
        'proprioception': {
            'joint_position': np.zeros(7, dtype=np.float64),
            'cartesian_position': np.zeros(6, dtype=np.float64),
            'gripper_position': np.zeros(1, dtype=np.float64),
        },
        'instruction': instruction,
    }
    if reward is not None:
        row['reward'] = reward
    return row


def _droid_policy(row):
    """Trivial policy: zero action, echo the instruction in the event meta."""
    from brainscore_core.streaming import StreamEvent
    return {'motor': StreamEvent('motor', np.zeros(7, dtype=np.float64), None,
                                 {'instruction_seen': row['instruction']})}


def _make_model(policy=None):
    """A minimal BrainScoreModel for closed-loop dispatch testing."""
    return BrainScoreModel(
        identifier='test-embodied-policy',
        model=None,
        region_layer_map={},
        in_channels={'vision:image'},
        policy=policy,
        policy_in_channels=('vision:image', 'proprioception', 'instruction'),
    )


# -- Dispatch -----------------------------------------------------------------

class TestProcessRow:

    def test_process_routes_to_policy(self):
        model = _make_model(policy=_droid_policy)
        outputs = model.process(_droid_row(instruction="pour the water"))
        assert set(outputs) == {'motor'}
        assert outputs['motor'].payload.shape == (7,)
        assert outputs['motor'].meta['instruction_seen'] == "pour the water"

    def test_process_no_policy_raises(self):
        model = _make_model(policy=None)
        with pytest.raises(NotImplementedError, match=r"BrainScoreModel\(policy=\.\.\.\)"):
            model.process(_droid_row())

    def test_process_policy_must_return_a_mapping(self):
        """A policy that returns a bare ndarray (instead of {'motor': action})
        raises a clear TypeError."""
        model = _make_model(policy=lambda row: np.zeros(7))
        with pytest.raises(TypeError, match="mapping of output channel"):
            model.process(_droid_row())

    def test_policy_receives_the_full_row(self):
        """The policy sees the unmodified row: cameras, body state, instruction."""
        captured = {}

        def capturing_policy(row):
            captured['cameras'] = list(row['vision:image'])
            captured['joint_pos'] = row['proprioception']['joint_position'].copy()
            captured['instruction'] = row['instruction']
            captured['reward'] = row['reward']
            return {'motor': np.zeros(7)}

        row = _droid_row(instruction="hello world", reward=0.25)
        row['proprioception']['joint_position'] = np.arange(7, dtype=np.float64)
        _make_model(policy=capturing_policy).process(row)

        assert captured['cameras'] == ['exterior_1', 'exterior_2', 'wrist']
        assert np.array_equal(captured['joint_pos'], np.arange(7))
        assert captured['instruction'] == "hello world" and captured['reward'] == 0.25


class TestMultiStepRollout:
    """Proxy for an embodied benchmark loop: process(row) repeatedly."""

    def test_rollout_returns_one_output_mapping_per_row(self):
        model = _make_model(policy=_droid_policy)
        responses = [model.process(_droid_row("reach the goal", reward=float(i == 4)))
                     for i in range(5)]
        assert len(responses) == 5
        assert all('motor' in r for r in responses)
