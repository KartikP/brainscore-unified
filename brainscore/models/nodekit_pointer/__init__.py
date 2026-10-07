"""Pointer policies for nodekit benchmarks, registered as scorable models.

- ``nodekit-random-pointer``: the null. Wanders and clicks at random without
  looking; whatever it scores is the floor.
- ``nodekit-straight-reach``: reference mover. Reads the privileged card layout
  and moves in a straight line at constant speed. Checks the harness end to
  end; it is not a model of people.
"""
from brainscore import model_registry
from brainscore_core.model_interface import BrainScoreModel
from brainscore.harnesses.nodekit_browser import random_pointer_policy, straight_reach_policy
from brainscore.model_helpers.policy_wrapper import PolicyWrapper


def pointer_model(identifier, policy):
    return BrainScoreModel(identifier, None, {}, {}, None,
                           action_fn=PolicyWrapper(policy, max_history=0))


model_registry['nodekit-random-pointer'] = lambda: pointer_model(
    'nodekit-random-pointer', random_pointer_policy(seed=0))
model_registry['nodekit-straight-reach'] = lambda: pointer_model(
    'nodekit-straight-reach', straight_reach_policy())
