"""Gymnasium environment harness — drive a real, standard RL env through the same
closed-loop ``process(EnvironmentStep)`` interface as the bespoke grid game.

This generalizes the embodied path from the toy ``GridGameEnv`` to any Gymnasium
env that renders RGB frames and has a discrete action space. The default target is
**MiniGrid** (turn-based, so a slow VLM policy isn't punished by real-time physics;
genuinely complex — key/door/multi-room planning; ships a natural-language mission
string). A VLM looks at the rendered frame, reads the mission + the legal-action
menu, and returns an action index — exactly the embodied schema, now on a standard
benchmark environment.

    from brainscore.harnesses.gymnasium_harness import play_gym_episode, MINIGRID_ACTIONS
    res = play_gym_episode(model, 'MiniGrid-DoorKey-6x6-v0', max_steps=60)

Needs ``gymnasium`` and (for MiniGrid) ``minigrid`` installed.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from brainscore_core.model_interface import EnvironmentResponse, EnvironmentStep

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


def play_gym_episode(model, env_id: str, max_steps: int = 60, seed: int = 0,
                     action_descriptions: Optional[Dict[int, str]] = None,
                     n_actions: Optional[int] = None) -> Dict[str, Any]:
    """Run one episode of ``env_id`` through ``model.process(EnvironmentStep)``.

    The observation handed to the model is ``{'frame', 'instruction', 'legal_actions'}``
    where ``frame`` is the full RGB render (god's-eye, agent included) and
    ``legal_actions`` is the index→description menu. Returns solved / steps /
    total_reward / the action trace + the mission string.
    """
    import gymnasium as gym
    if 'MiniGrid' in env_id:
        import minigrid  # noqa: F401  registers the MiniGrid-* envs with gymnasium
    env = gym.make(env_id, render_mode='rgb_array')
    try:
        obs, info = env.reset(seed=seed)
        mission = _mission(env, obs)
        if action_descriptions is None:
            action_descriptions = (MINIGRID_ACTIONS if 'MiniGrid' in env_id
                                   else {i: f'action {i}' for i in range(int(env.action_space.n))})
        n_actions = n_actions or int(env.action_space.n)

        actions: List[int] = []
        total_reward, solved = 0.0, False
        for t in range(max_steps):
            frame = env.render()                       # (H, W, 3) uint8 RGB
            step = EnvironmentStep(
                observation={'frame': np.asarray(frame), 'instruction': mission,
                             'legal_actions': dict(action_descriptions)},
                instruction=mission, is_first=(t == 0), step_num=t)
            response = model.process(step)
            if not isinstance(response, EnvironmentResponse):
                raise TypeError(f'policy returned {type(response).__name__}, expected EnvironmentResponse')
            action = int(np.asarray(response.action).reshape(-1)[0]) % n_actions
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                solved = bool(terminated and reward > 0)   # MiniGrid: positive reward only on success
                break
        return {'env_id': env_id, 'seed': seed, 'mission': mission, 'solved': solved,
                'steps': len(actions), 'total_reward': round(total_reward, 4),
                'actions': actions, 'optimal_unknown': True}
    finally:
        env.close()


def random_gym_policy(seed: int = 0):
    """Uniform-random discrete policy — the embodied null floor for a Gym env."""
    rng = np.random.RandomState(seed)

    def policy(observation, history):
        n = len(observation.get('legal_actions', {})) or 7
        return int(rng.randint(0, n))
    return policy
