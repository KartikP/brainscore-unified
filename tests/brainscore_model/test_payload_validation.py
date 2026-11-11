"""Core is device-agnostic: it passes closed-loop observations to the policy
without inspecting them. Robot camera checks live in the robotics harness."""
import numpy as np

from brainscore import BrainScoreModel


def _agent(seen):
    return BrainScoreModel('agent', None, {}, {}, None,
                           policy=lambda row: seen.append(row) or {'motor': np.zeros(7)})


def test_core_passes_device_payloads_to_the_policy_unchecked():
    seen = []
    # A malformed frame core used to reject now reaches the policy untouched.
    row = {'observation': {'cameras': {'wrist': np.zeros((2, 2))}}}
    assert set(_agent(seen).process(row)) == {'motor'}
    assert seen == [row]
