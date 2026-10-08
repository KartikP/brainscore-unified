"""Turn-based Gymnasium evaluation with RGB observations and discrete actions."""
from typing import Any, Dict, List, Optional

import numpy as np

from brainscore_core.model_interface import EnvironmentStep
from brainscore_core.environment import ArraySpec, DiscreteSpec, MappingSpec, TextSpec
from brainscore_core.streaming_helpers import run_environment

# MiniGrid's discrete action set (Actions enum), described for a language policy.
MINIGRID_ACTIONS: Dict[int, str] = {
    0: 'turn left',
    1: 'turn right',
    2: 'move forward',
    3: 'pick up the object in front',
    4: 'drop the carried object',
    5: 'toggle / open (a door, with a key if locked)',
    6: 'done (declare the task complete)',
}


def _mission(env, obs) -> str:
    if isinstance(obs, dict) and obs.get('mission'):
        return str(obs['mission'])
    return str(getattr(getattr(env, 'unwrapped', env), 'mission', '') or '')


class GymnasiumEnvironment:
    """Expose a rendered Gymnasium environment as a UMI environment wrapper.

    Observations are full RGB renders, instructions and legal actions, not the
    environment's native observation_space. The caller supplies the environment;
    close() releases it. Success scoring remains outside this wrapper.
    """

    def __init__(self, env, *, max_steps=60, action_descriptions=None):
        import gymnasium as gym
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError('GymnasiumEnvironment requires discrete actions')
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        self.env = env
        self.max_steps = max_steps
        self._action_spec = DiscreteSpec(int(env.action_space.n), int(env.action_space.start))
        self.action_descriptions = dict(action_descriptions) if action_descriptions is not None else {
            i: f'action {i}' for i in range(
                self._action_spec.start, self._action_spec.start + self._action_spec.count
            )
        }
        expected = set(range(self._action_spec.start, self._action_spec.start + self._action_spec.count))
        if set(self.action_descriptions) != expected:
            raise ValueError('Action descriptions must cover the declared action space')
        self.steps = 0
        self.total_reward = 0.
        self.terminated = self.truncated = False
        self.last_reward = 0.
        self.mission = ''
        self.last_info = {}

    def observation_spec(self):
        return MappingSpec({
            'frame': ArraySpec((None, None, 3), 'uint8'),
            'instruction': TextSpec(),
            'legal_actions': MappingSpec({key: TextSpec() for key in self.action_descriptions}),
        })

    def action_spec(self):
        return self._action_spec

    def _observation(self, obs):
        self.mission = _mission(self.env, obs)
        return {
            'frame': np.asarray(self.env.render()),
            'instruction': self.mission,
            'legal_actions': dict(self.action_descriptions),
        }

    def reset(self, *, seed=None, options=None):
        obs, self.last_info = self.env.reset(seed=seed, options=options)
        self.steps = 0
        self.total_reward = self.last_reward = 0.
        self.terminated = self.truncated = False
        observation = self._observation(obs)
        return EnvironmentStep(
            observation=observation, instruction=self.mission,
            is_first=True, step_num=0,
        )

    def step(self, action):
        action = self._action_spec.validate(action)
        obs, reward, terminated, truncated, self.last_info = self.env.step(action)
        self.steps += 1
        self.last_reward = float(reward)
        self.total_reward += self.last_reward
        self.terminated = bool(terminated)
        self.truncated = bool(truncated or (self.steps >= self.max_steps and not terminated))
        observation = self._observation(obs)
        return EnvironmentStep(
            observation=observation, instruction=self.mission, step_num=self.steps,
            reward=self.last_reward, is_terminal=self.terminated,
            is_last=self.terminated or self.truncated,
        )

    def close(self):
        self.env.close()


def play_gym_episode(model, env_id: str, max_steps: int = 60, seed: int = 0,
                     action_descriptions: Optional[Dict[int, str]] = None,
                     n_actions: Optional[int] = None) -> Dict[str, Any]:
    """Run a rendered, discrete Gymnasium episode through EnvironmentSession.

    MiniGrid uses terminal positive reward as success. Other environments must
    provide info['is_success']; otherwise solved is None (unknown). The caller
    owns subject reset, or can use SessionProtocol with GymnasiumEnvironment.
    """
    import gymnasium as gym
    if 'MiniGrid' in env_id:
        import minigrid  # noqa: F401 -- registers MiniGrid environments.
    env = gym.make(env_id, render_mode='rgb_array')
    try:
        if n_actions is not None and n_actions != int(env.action_space.n):
            raise ValueError('n_actions must match the environment action space')
        descriptions = action_descriptions
        if descriptions is None and 'MiniGrid' in env_id:
            descriptions = MINIGRID_ACTIONS
        wrapper = GymnasiumEnvironment(
            env, max_steps=max_steps, action_descriptions=descriptions,
        )
    except BaseException:
        env.close()
        raise
    responses = run_environment(model, wrapper, seed=seed, require_specs=True)
    solved = (bool(wrapper.terminated and wrapper.last_reward > 0) if 'MiniGrid' in env_id
              else wrapper.last_info.get('is_success'))
    return {
        'env_id': env_id, 'seed': seed, 'mission': wrapper.mission,
        'solved': None if solved is None else bool(solved),
        'steps': len(responses), 'total_reward': round(wrapper.total_reward, 4),
        'actions': [int(np.asarray(response.action).item()) for response in responses],
        'optimal_unknown': True,
    }


def random_gym_policy(seed: int = 0):
    """Uniform-random discrete policy — the embodied null floor for a Gym env."""
    rng = np.random.RandomState(seed)

    def policy(observation, history):
        n = len(observation.get('legal_actions', {})) or 7
        return int(rng.randint(0, n))
    return policy
