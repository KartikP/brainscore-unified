"""Core is device-agnostic: it passes closed-loop rows to the policy and the
policy's action payloads back without inspecting them. Robot camera checks live
in the robotics harness."""
import numpy as np

from brainscore import BrainScoreModel


def _agent(seen):
    return BrainScoreModel('agent',
                           policy=lambda row: seen.append(row) or {'motor': np.zeros(7)})


def test_core_passes_device_payloads_to_the_policy_unchecked():
    seen = []
    # A malformed frame core used to reject now reaches the policy untouched.
    row = {'observation': {'cameras': {'wrist': np.zeros((2, 2))}}}
    assert set(_agent(seen).process(row)) == {'motor'}
    assert seen == [row]


def test_the_action_payload_is_harness_defined():
    # A non-DROID action payload (a discrete game action, a vector) is fine.
    model = BrainScoreModel('discrete', policy=lambda row: {'motor': 'LEFT'})
    assert model.process({'observation': {}}) == {'motor': 'LEFT'}
    model = BrainScoreModel('vector', policy=lambda row: {'motor': np.zeros(7)})
    assert model.process({'observation': {}})['motor'].shape == (7,)
