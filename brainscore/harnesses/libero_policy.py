"""Preserve OpenPI LIBERO requests and action chunks across the UMI boundary.

The reference evaluator owns image transforms, action scheduling, and robot
control. This adapter accepts its already prepared request. It does not reuse
DROID's joint-velocity action definition or apply additional transforms.
"""
from copy import deepcopy

import numpy as np

from brainscore_core.events import EnvironmentResponse, EnvironmentStep


class LiberoChunkPolicy:
    """Adapt ``infer(request)`` to a UMI action provider without consuming chunks.

    The seven outputs are the LIBERO controller's six end-effector commands
    and gripper command, in the checkpoint's convention. They are not seven
    joint velocities. Controller scaling remains in the reference environment.
    """

    def __init__(self, policy, *, execution_horizon=None):
        if not callable(getattr(policy, 'infer', None)):
            raise TypeError('Policy must expose infer(request)')
        if execution_horizon is not None and (
                type(execution_horizon) is not int or execution_horizon < 1):
            raise ValueError('execution_horizon must be a positive integer')
        self.policy = policy
        self.execution_horizon = execution_horizon

    def __call__(self, step):
        if not isinstance(step, EnvironmentStep):
            raise TypeError('Expected EnvironmentStep')
        observation = step.observation
        request = {}
        for name in ('observation/image', 'observation/wrist_image'):
            image = np.asarray(observation[name])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f'{name} must be an H x W x 3 uint8 image')
            request[name] = image.copy()
        state = np.asarray(observation['observation/state'])
        if state.shape != (8,) or state.dtype.kind not in 'fiu' or not np.isfinite(state).all():
            raise ValueError('Expected eight finite LIBERO state values')
        request['observation/state'] = state.copy()
        if not isinstance(step.instruction, str):
            raise ValueError('Expected a text instruction')
        request['prompt'] = step.instruction
        result = self.policy.infer(request)
        actions = np.asarray(result['actions'])
        if (actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 7
                or actions.dtype.kind not in 'fiu' or not np.isfinite(actions).all()):
            raise ValueError('Expected a nonempty finite action chunk with shape (horizon, 7)')
        if self.execution_horizon is not None and self.execution_horizon > len(actions):
            raise ValueError('Predicted chunk does not cover execution_horizon')
        return EnvironmentResponse(
            action=actions.copy(),
            metadata={
                'policy_output': deepcopy({k: v for k, v in result.items() if k != 'actions'}),
                'action_kind': 'chunk', 'prediction_horizon': len(actions),
                'execution_horizon': self.execution_horizon,
                'scheduler': 'external_evaluator',
            },
        )
