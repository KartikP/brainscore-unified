"""Connect a caller-supplied OpenPI-style policy to the UMI action provider."""
import numpy as np

from brainscore_core.events import EnvironmentResponse, EnvironmentStep


class DroidPolicy:
    """Map DROID observations to ``policy.infer(request)['actions']``.

    No weights, network client, hardware, or image transforms are created here.
    The supplied policy owns preprocessing and inference. ``reset_policy`` must
    restore the state required by the experiment, including RNG state if exact
    paired comparisons are intended. ``action_spec`` describes the eight policy
    outputs; its units and bounds must match the checkpoint and chosen protocol.
    """

    def __init__(self, policy, *, action_spec, reset_policy, external_camera='exterior_1',
                 action_horizon=1):
        if external_camera not in ('exterior_1', 'exterior_2'):
            raise ValueError('Choose exterior_1 or exterior_2 explicitly')
        if type(action_horizon) is not int or action_horizon < 1:
            raise ValueError('action_horizon must be a positive integer')
        if len(action_spec.names) != 8:
            raise ValueError('DROID policy requires seven joint commands and one gripper command')
        if not callable(reset_policy):
            raise TypeError('Provide an explicit policy reset callback')
        if not callable(getattr(policy, 'infer', None)):
            raise TypeError('Policy must expose infer(request)')
        self.policy = policy
        self.action_spec = action_spec
        self.reset_policy = reset_policy
        self.external_camera = external_camera
        self.action_horizon = action_horizon
        self._pending = []
        self._next_step = None

    def reset(self):
        self._pending.clear()
        self._next_step = None
        self.reset_policy()

    def __call__(self, step):
        if not isinstance(step, EnvironmentStep):
            raise TypeError('DroidPolicy expects an EnvironmentStep')
        if step.is_first:
            self.reset()
        if self._next_step is not None and step.step_num != self._next_step:
            self._pending.clear()
            self._next_step = None
            raise ValueError('Steps must be consecutive; reset before starting another episode')
        if not self._pending:
            cameras = step.observation['cameras']
            body = step.observation['proprioception']
            request = {
                'observation/exterior_image_1_left': np.array(cameras[self.external_camera].rgb, copy=True),
                'observation/wrist_image_left': np.array(cameras['wrist'].rgb, copy=True),
                'observation/joint_position': np.array(body.joint_position, copy=True),
                'observation/gripper_position': np.array(body.gripper_position, copy=True),
                'prompt': step.instruction or '',
            }
            try:
                actions = np.asarray(self.policy.infer(request)['actions'])
                if actions.ndim != 2 or actions.shape[1] != 8 or len(actions) < self.action_horizon:
                    raise ValueError('Expected an action chunk of shape (horizon, 8) covering action_horizon')
                checked = [self.action_spec.validate(action) for action in actions]
            except BaseException:
                self._pending.clear()
                self._next_step = None
                raise
            self._pending = checked[:self.action_horizon]
        response = EnvironmentResponse(action=self._pending.pop(0), metadata={
            'action_horizon': self.action_horizon, 'external_camera': self.external_camera,
            'step_num': step.step_num, 'action_transform': 'none'})
        self._next_step = step.step_num + 1
        if step.is_last:
            self._pending.clear()
            self._next_step = None
        return response
