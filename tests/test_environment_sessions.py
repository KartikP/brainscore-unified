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
            GridGameEnvironment(GridGameEnv(size=4)), seed=trial.identifier, require_specs=True,
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


@pytest.mark.parametrize('history_limit', [0, 2])
def test_record_episode_without_collecting_retained_history(tmp_path, history_limit):
    sessions = []
    @contextmanager
    def factory(trial):
        with EnvironmentSession(
            GridGameEnvironment(GridGameEnv(), max_steps=20),
            seed=7,
            require_specs=True,
            history_limit=history_limit,
        ) as session:
            sessions.append(session)
            yield session
    result = Experiment(
        subject=subject(),
        protocol=SessionProtocol(
            'record-episode', factory,
            input_channels=('observation',),
            output_channels=('motor',),
            collect_outputs=False,
        ),
        tools=[RecordInputsOutputs(max_bytes=2_000_000)],
        output_dir=tmp_path / 'episode',
    ).run()
    assert result.value is None and result.manifest['status'] == 'complete'
    assert result.manifest['plan']['protocol']['config']['collect_outputs'] is False
    session, = sessions
    assert session.complete
    assert len(session.input_events) <= history_limit
    assert len(session.emitted) <= history_limit
    assert len(session.action_events) <= history_limit
    events = list(result.record.events())
    applied = [e for e in events if e['kind'] == 'output'
               and e['payload'].meta.get('action_status') == 'applied']
    proposed = [e for e in events if e['kind'] == 'output'
                and e['payload'].meta.get('action_status') == 'proposed']
    inputs = [e for e in events if e['kind'] == 'input']
    assert len(applied) == len(proposed) == len(inputs) - 1
    assert len(applied) > len(session.emitted)
    with pytest.raises(RuntimeError, match='history_limit'):
        session.collect()


@pytest.mark.parametrize('value', [0, 1, None, 'false'])
def test_collection_setting_rejects_ambiguous_values(value):
    with pytest.raises(TypeError, match='collect_outputs'):
        SessionProtocol('invalid', lambda trial: None, collect_outputs=value)

def test_portable_external_environment_check(tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1] / 'examples/external_environment_checks/run.py'
    spec = importlib.util.spec_from_file_location('external_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run(tmp_path / 'qualification', steps=3)
    results = report['conditions']
    assert results['baseline']['action_sha256'] == results['restored']['action_sha256']
    assert results['scaled']['activity_events_verified'] == 3


def test_portable_check_saves_failure_report(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    source = Path(__file__).parents[1] / 'examples/external_environment_checks/run.py'
    spec = importlib.util.spec_from_file_location('external_check_failure', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def fail(*args, **kwargs):
        raise RuntimeError('Injected qualification failure')
    monkeypatch.setattr(module, 'check_repeated_failure', fail)
    with pytest.raises(RuntimeError, match='Injected qualification failure'):
        module.run(tmp_path / 'qualification', steps=2)
    report = json.loads((tmp_path / 'qualification' / 'report.json').read_text())
    assert report['status'] == 'failed'
    assert report['error_type'] == 'RuntimeError'


def test_portable_check_gates_survive_optimized_python():
    import os
    from pathlib import Path
    import subprocess
    import sys
    source = Path(__file__).parents[1] / 'examples/external_environment_checks/run.py'
    code = (
        "import runpy; "
        f"check = runpy.run_path({str(source)!r}); "
        "check['require'](False, 'qualification gate')"
    )
    result = subprocess.run(
        [sys.executable, '-O', '-c', code], capture_output=True, text=True,
        timeout=30, env=dict(os.environ),
    )
    assert result.returncode != 0
    assert 'RuntimeError: qualification gate' in result.stderr


@pytest.fixture
def portable_check():
    import importlib.util
    from pathlib import Path
    source = Path(__file__).parents[1] / 'examples/external_environment_checks/run.py'
    spec = importlib.util.spec_from_file_location('portable_lifecycle', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portable_provenance_names_imported_code(portable_check):
    import brainscore
    import brainscore_core
    from pathlib import Path
    modules = portable_check.provenance()['modules']
    assert modules['brainscore']['imported_file'] == str(Path(brainscore.__file__).resolve())
    assert modules['brainscore-core']['imported_file'] == str(Path(brainscore_core.__file__).resolve())


def test_portable_never_overwrites_existing_run(tmp_path, portable_check):
    directory = tmp_path / 'existing'
    directory.mkdir()
    report = directory / 'report.json'
    report.write_text('keep')
    with pytest.raises(FileExistsError):
        portable_check.run(directory, steps=2)
    assert report.read_text() == 'keep'


def test_portable_two_real_worker_processes(tmp_path, portable_check):
    import json
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, '-O', portable_check.__file__, '--out', str(tmp_path / 'workers'),
         '--steps', '2', '--workers', '2'], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    reports = json.loads((tmp_path / 'workers' / 'workers.json').read_text())
    assert len(reports) == 2 and all(r['status'] == 'complete' for r in reports)
    for index in (0, 1):
        assert (tmp_path / 'workers' / f'worker-{index}' / 'recording').is_dir()


def test_portable_failed_worker_is_reported(tmp_path, portable_check, monkeypatch):
    import json
    import sys
    class FailedWorker:
        def wait(self, **kwargs):
            return 1
        def poll(self):
            return 1
    monkeypatch.setattr(portable_check.subprocess, 'Popen', lambda *a, **k: FailedWorker())
    monkeypatch.setattr(sys, 'argv', ['run.py', '--out', str(tmp_path / 'workers'), '--workers', '2'])
    with pytest.raises(RuntimeError, match='Worker failures'):
        portable_check.main()
    report = json.loads((tmp_path / 'workers' / 'report.json').read_text())
    assert report['status'] == 'failed' and report['return_codes'] == [1, 1]


def test_portable_has_no_disabled_assert_gates(portable_check):
    import ast
    from pathlib import Path
    tree = ast.parse(Path(portable_check.__file__).read_text())
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def test_source_state_reports_a_commit_only_for_clean_tracked_code(tmp_path, portable_check):
    import subprocess
    def git(*args):
        subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=t', '-c', 'user.email=t@t',
                        '-c', 'commit.gpgsign=false', *args], check=True, capture_output=True)
    module = tmp_path / 'module.py'
    module.write_text('x = 1\n')
    git('init', '-q')
    git('add', 'module.py')
    git('commit', '-q', '-m', 'fixture')
    head = subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], text=True).strip()
    state = portable_check.source_state
    assert state(module) == {'source_commit': head, 'git_head': head, 'source_dirty': False}
    (tmp_path / 'untracked.py').write_text('y = 2\n')
    assert state(module) == {'source_commit': None, 'git_head': head, 'source_dirty': True}
    (tmp_path / 'untracked.py').unlink()
    module.write_text('x = 2\n')
    assert state(module) == {'source_commit': None, 'git_head': head, 'source_dirty': True}
    assert state(tmp_path / 'new.py') == {'source_commit': None, 'git_head': head, 'source_dirty': None}
    outside = tmp_path.parent / f'{tmp_path.name}-outside'
    outside.mkdir()
    assert state(outside / 'module.py') == {'source_commit': None, 'git_head': None,
                                             'source_dirty': None}


def test_recorded_instrumented_episode_keeps_no_per_step_objects(tmp_path, portable_check):
    """Count live traced blocks after a full collection: retaining anything per step adds blocks.

    Raw bytes and uncollected block counts are the wrong signal. CPython reallocates its
    process-wide interned-string table (several MB) when it runs out of slots, and keeps
    bounded freelists and not-yet-collected cycles until a full collection.
    """
    from contextlib import contextmanager
    import gc
    import tracemalloc
    import numpy as np
    import torch
    from brainscore.experiments import (Experiment, RecordActivity, RecordInputsOutputs,
                                        ScaleActivity, TorchInstrumentation)
    from brainscore_core.events import EnvironmentResponse
    from brainscore_core.model_interface import BrainScoreModel
    from brainscore_core.streaming_helpers import EnvironmentSession
    if tracemalloc.is_tracing():
        pytest.skip('tracemalloc already active')
    layer = torch.nn.Linear(1, 1, bias=False)
    marks, blocks, calls = (300, 1500), {}, [0]

    def policy(step):
        calls[0] += 1
        if calls[0] in marks:
            gc.collect()
            blocks[calls[0]] = len(tracemalloc.take_snapshot().traces)
        with torch.no_grad():
            return EnvironmentResponse(action=layer(torch.from_numpy(step.observation['state'])).numpy().copy())

    environment = portable_check.CounterEnvironment(marks[-1])

    @contextmanager
    def factory(trial):
        with EnvironmentSession(environment, require_specs=True, history_limit=0) as session:
            yield session
    tracemalloc.start()
    try:
        Experiment(subject=BrainScoreModel(identifier='fixture', policy=policy),
                   protocol=SessionProtocol('episode', factory, input_channels=('observation',),
                                            output_channels=('motor',), collect_outputs=False),
                   tools=[RecordInputsOutputs(), ScaleActivity(['policy'], factor=0.5),
                          RecordActivity(['policy'])],
                   instrumentation=TorchInstrumentation({'policy': layer}),
                   output_dir=tmp_path / 'run').run()
    finally:
        tracemalloc.stop()
    # 1,200 recorded, scaled and observed steps; one retained object per step would add >= 1,200.
    assert blocks[1500] - blocks[300] < 50, blocks
