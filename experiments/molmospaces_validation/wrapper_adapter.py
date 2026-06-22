"""UnifiedInterfacePolicy — wraps PI_Policy through our EnvironmentStep schema.

Validates that the BrainScoreModel(action_fn=...) + EnvironmentStep schema is
end-to-end lossless on a real DROID-trained policy in the MolmoSpaces simulator.

How the wrapper differs from stock PI_Policy:

- Stock:    raw_obs → PI_Policy.obs_to_model_input → π0 dict → server → action
- Wrapped:  raw_obs → EnvironmentStep (OUR schema) → action_fn:
                          → env_step_to_pi_input → π0 dict → server → action
                      → EnvironmentResponse → action

Both runs hit the SAME π0 server with the SAME inputs IF our schema is lossless.
Score divergence implies the schema is dropping information.

This file lives both locally (under version control) AND on EC2 in the eval
pipeline. The EC2 copy is what actually runs.
"""

import os
import time
from typing import Any, Dict

import numpy as np

# These imports come from the molmospaces env on EC2:
from molmo_spaces.policy.learned_policy.pi_policy import PI_Policy
from molmo_spaces.policy.learned_policy.utils import resize_with_pad

# These come from brain-score-unified (editable install on EC2):
from brainscore_core.model_interface import (
    BrainScoreModel,
    CameraFrame,
    EnvironmentResponse,
    EnvironmentStep,
    Proprioception,
)


def raw_obs_to_env_step(obs: Dict[str, Any], instruction: str,
                         step_num: int = 0) -> EnvironmentStep:
    """Convert molmospaces raw obs dict into our DROID-shaped EnvironmentStep.

    Schema mapping:
      obs['wrist_camera']      → cameras['wrist'].rgb
      obs['exo_camera_1']      → cameras['exterior_1'].rgb
      obs['qpos']['arm']       → proprioception.joint_position
      obs['qpos']['gripper']   → proprioception.gripper_position (clipped to [0,1])
      task.get_task_description() → instruction

    Note: molmospaces sometimes uses alternate keys (wrist_camera_zed_mini,
    droid_shoulder_light_randomization). We mirror PI_Policy's fallback logic.
    """
    if isinstance(obs, (list, tuple)):
        obs = obs[0]

    exo_key = ('droid_shoulder_light_randomization'
               if 'droid_shoulder_light_randomization' in obs else 'exo_camera_1')
    wrist_key = ('wrist_camera_zed_mini'
                 if 'wrist_camera_zed_mini' in obs else 'wrist_camera')

    cameras = {
        'exterior_1': CameraFrame(rgb=np.asarray(obs[exo_key])),
        'wrist': CameraFrame(rgb=np.asarray(obs[wrist_key])),
    }

    grip_norm = float(np.clip(obs['qpos']['gripper'][0] / 0.824033, 0.0, 1.0))
    proprio = Proprioception(
        joint_position=np.asarray(obs['qpos']['arm'][:7]).reshape(7),
        cartesian_position=np.zeros(6, dtype=np.float64),  # not provided in obs
        gripper_position=np.asarray([grip_norm]),
    )

    return EnvironmentStep(
        cameras=cameras,
        proprioception=proprio,
        instruction=instruction.lower() if instruction else None,
        step_num=step_num,
    )


def env_step_to_pi_input(env_step: EnvironmentStep) -> Dict[str, Any]:
    """Convert our EnvironmentStep back into the dict format the π0 server
    expects. The pad-to-224 resize happens here to match stock PI_Policy."""
    return {
        'observation/exterior_image_1_left': resize_with_pad(
            env_step.cameras['exterior_1'].rgb, 224, 224),
        'observation/wrist_image_left': resize_with_pad(
            env_step.cameras['wrist'].rgb, 224, 224),
        'observation/joint_position': np.asarray(
            env_step.proprioception.joint_position).reshape(7),
        'observation/gripper_position': np.asarray(
            env_step.proprioception.gripper_position).reshape(1),
        'prompt': env_step.instruction or '',
    }


class UnifiedInterfacePolicy(PI_Policy):
    """PI_Policy that routes obs/action through our EnvironmentStep schema.

    All π0 inference still goes to the same openpi server (default localhost:8080);
    the only thing this wrapper changes is the data shape carrying obs/action
    between the molmospaces eval driver and the π0 server.
    """

    def __init__(self, exp_config, task=None) -> None:
        super().__init__(exp_config, task)
        self._step_counter = 0

        # Build the BrainScoreModel that owns the schema-bridging action_fn.
        # The action_fn receives our EnvironmentStep and returns an
        # EnvironmentResponse.
        def _action_fn(env_step: EnvironmentStep) -> EnvironmentResponse:
            pi_input = env_step_to_pi_input(env_step)
            action = self._infer_pi_buffered(pi_input)
            return EnvironmentResponse(action=action)

        self._brain_score_model = BrainScoreModel(
            identifier='pi0-fast-droid-via-umi',
            model=None,
            region_layer_map={},
            preprocessors={'vision': lambda x: x},
            action_fn=_action_fn,
        )

    def reset(self) -> None:
        super().reset()
        self._step_counter = 0

    def obs_to_model_input(self, obs):
        """Convert raw molmospaces obs → our EnvironmentStep. The schema bridge
        happens here (input side)."""
        instruction = self.task.get_task_description()
        env_step = raw_obs_to_env_step(obs, instruction, self._step_counter)
        self._step_counter += 1
        return env_step

    def inference_model(self, env_step: EnvironmentStep) -> np.ndarray:
        """Route through our BrainScoreModel — exercises process(EnvironmentStep)
        dispatch + action_fn + EnvironmentResponse."""
        env_response = self._brain_score_model.process(env_step)
        return env_response.action

    def model_output_to_action(self, model_output):
        """Reuse stock PI_Policy split-into-arm+gripper logic — the schema
        bridge ends one layer up at inference_model output."""
        return super().model_output_to_action(model_output)

    def _infer_pi_buffered(self, pi_input: Dict[str, Any]) -> np.ndarray:
        """Replicate PI_Policy.inference_model's chunk-buffering websocket call
        but factored out so it can live inside the action_fn closure."""
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()
        if self.actions_buffer is None or self.current_buffer_index >= self.chunk_size:
            import websockets
            try:
                self.actions_buffer = self.model.infer(pi_input)['actions']
            except websockets.exceptions.ConnectionClosedError:
                self.prepare_model()
                time.sleep(5)
                self.actions_buffer = self.model.infer(pi_input)['actions']
            self.current_buffer_index = 0
        action = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return action
