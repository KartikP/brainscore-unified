"""Native environment parity and tool records; no trained-policy qualification."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from brainscore_core.environment import ActionSpec
from brainscore_core.events import EnvironmentResponse, EnvironmentStep
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.streaming_helpers import EnvironmentSession, run_environment
from brainscore.experiments import Experiment, SessionProtocol, RecordInputsOutputs
from brainscore.experiments.replay import read_events, replay_sessions
from brainscore.harnesses.grid_game import GridGameEnv, GridGameEnvironment, greedy_oracle_policy, play_game
from brainscore.harnesses.gymnasium_harness import GymnasiumEnvironment, play_gym_episode
from brainscore.model_helpers.policy_wrapper import PolicyWrapper


def subject(policy=greedy_oracle_policy):
    return BrainScoreModel('grid', action_fn=PolicyWrapper(policy))


def test_action_spec_is_one_shared_definition():
    from brainscore.robotics import ActionSpec as RoboticsActionSpec
    assert RoboticsActionSpec is ActionSpec


@pytest.mark.parametrize('seed', [0, 7, 21])
def test_grid_session_preserves_reference_actions_rewards_and_score(seed):
    # Independent direct loop: no session or wrapper drives this reference.
    env = GridGameEnv(seed=seed)
    obs = env.reset()
    distance = env._manhattan()
    actions, total = [], 0.
    while True:
        action = greedy_oracle_policy(obs, ())
        obs, reward, done, info = env.step(action)
        actions.append(action)
        total += reward
        if done:
            break
    result = play_game(subject(), GridGameEnv(seed=seed))
    assert result['actions'] == actions
    assert result['total_reward'] == round(total, 4)
    assert result['solved'] == info['reached_goal']
    assert result['steps'] == len(actions) == distance


def test_grid_limit_is_truncation_and_reset_seed_reproduces_start():
    wrapper = GridGameEnvironment(GridGameEnv(), max_steps=1)
    start = wrapper.reset(seed=12)
    start_again = wrapper.reset(seed=12)
    np.testing.assert_array_equal(start.observation['frame'], start_again.observation['frame'])
    result = wrapper.step(0)
    assert result.is_last
    assert result.is_terminal == wrapper.solved


def test_gym_session_preserves_seeded_direct_reference():
    import gymnasium as gym
    import minigrid  # noqa: F401 -- environment registration.
    env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='rgb_array')
    rng = np.random.RandomState(1)
    actions, reward_sum = [], 0.
    try:
        env.reset(seed=7)
        for _ in range(10):
            action = int(rng.randint(7))
            _, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            reward_sum += reward
            if terminated or truncated:
                break
    finally:
        env.close()
    rng = np.random.RandomState(1)
    result = play_gym_episode(
        subject(lambda obs, history: int(rng.randint(7))),
        'MiniGrid-Empty-5x5-v0', max_steps=10, seed=7,
    )
    assert result['actions'] == actions
    assert result['total_reward'] == round(reward_sum, 4)


@pytest.mark.parametrize('action', [-1, 7, 0.5, [0, 1]])
def test_gym_invalid_action_never_steps_and_closes(action):
    import gymnasium as gym
    calls = []
    env = SimpleNamespace(
        action_space=gym.spaces.Discrete(7),
        reset=lambda **kwargs: ({'mission': 'test'}, {}),
        render=lambda: np.zeros((4, 4, 3), dtype=np.uint8),
        step=lambda action: calls.append('step'),
        close=lambda: calls.append('close'),
    )
    with pytest.raises(ValueError):
        run_environment(subject(lambda obs, history: action), GymnasiumEnvironment(env))
    assert calls == ['close']


def test_session_tools_record_applied_actions_and_replay_saved_observations(tmp_path):
    sessions = []
    @contextmanager
    def factory(trial):
        with EnvironmentSession(
            GridGameEnvironment(GridGameEnv(size=4)), seed=trial, require_specs=True,
        ) as session:
            sessions.append(session)
            yield session
    model = subject()
    result = Experiment(
        subject=model,
        protocol=SessionProtocol(
            'grid', factory, trials=(5,),
            input_channels={'observation'}, output_channels={'motor'},
        ),
        tools=[RecordInputsOutputs()], output_dir=tmp_path/'run',
    ).run()
    events = read_events(result.directory)
    inputs = [e['payload'] for e in events if e['kind'] == 'input']
    outputs = [e['payload'] for e in events if e['kind'] == 'output']
    applied = [e for e in outputs if e.meta['action_status'] == 'applied']
    proposed = [e for e in outputs if e.meta['action_status'] == 'proposed']
    assert len(inputs) == len(applied) + 1
    assert inputs[-1].is_terminal
    assert len(applied) == len(proposed) == len(sessions[0].emitted) > 0
    assert all(e.t_ms is None for e in outputs)
    assert not model._action_fn.history  # SessionProtocol reset after the trial.
    replay = Experiment(
        subject=subject(), protocol=replay_sessions(result.directory),
        tools=[RecordInputsOutputs()], output_dir=tmp_path/'replay',
    ).run()
    replay_outputs = [e['payload'] for e in read_events(replay.directory) if e['kind'] == 'output']
    assert [e.payload.action for e in replay_outputs] == [e.payload.action for e in proposed]
    assert all(e.meta['action_status'] == 'proposed' for e in replay_outputs)


def test_caught_environment_failure_cannot_complete_an_experiment(tmp_path):
    class Broken:
        def reset(self):
            return EnvironmentStep(observation={})
        def step(self, action):
            raise RuntimeError('connection lost')
    class Catches(BrainScoreModel):
        def interact(self, session):
            try:
                super().interact(session)
            except RuntimeError:
                pass
    @contextmanager
    def factory(trial):
        with EnvironmentSession(Broken()) as session:
            yield session
    with pytest.raises(RuntimeError, match='before episode completion'):
        Experiment(
            subject=Catches('broken', action_fn=lambda step: EnvironmentResponse(0)),
            protocol=SessionProtocol('broken', factory),
            tools=[RecordInputsOutputs()], output_dir=tmp_path/'broken',
        ).run()
    import json
    assert json.loads((tmp_path/'broken'/'experiment.json').read_text())['status'] == 'failed'


def test_early_subject_return_cannot_complete_an_episode(tmp_path):
    class Early(BrainScoreModel):
        def interact(self, session):
            session.next_input()
    @contextmanager
    def factory(trial):
        with EnvironmentSession(GridGameEnvironment(GridGameEnv())) as session:
            yield session
    with pytest.raises(RuntimeError, match='before episode completion'):
        Experiment(
            subject=Early('early', action_fn=lambda step: EnvironmentResponse(0)),
            protocol=SessionProtocol('grid', factory),
            tools=[RecordInputsOutputs()], output_dir=tmp_path/'early',
        ).run()
