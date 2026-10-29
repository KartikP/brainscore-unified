"""Offline contract checks; these are not trained-policy qualification."""
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from brainscore.harnesses.libero_policy import LiberoChunkPolicy
from brainscore.run_record import RunRecord, RunRecorder
from brainscore_core.events import EnvironmentStep


def example_module(name):
    path = Path(__file__).parents[1] / 'examples' / 'libero' / (name + '.py')
    spec = importlib.util.spec_from_file_location('libero_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request():
    return {'observation/image': np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
            'observation/wrist_image': np.full((4, 4, 3), 72, dtype=np.uint8),
            'observation/state': np.arange(8, dtype=np.float32),
            'prompt': 'put the cup in the bowl'}


class FixturePolicy:
    def __init__(self):
        self.requests = []

    def infer(self, observation):
        self.requests.append(observation)
        # Unique rows make truncation, reordering, and implicit clipping visible.
        actions = np.arange(70, dtype=np.float32).reshape(10, 7)
        return {'actions': actions, 'state': observation['observation/state'].copy(),
                'policy_timing': {'infer_ms': 0.1}}


def test_subject_route_preserves_reference_requests_and_full_action_chunk(tmp_path):
    bridge = example_module('serve_bridge')
    results = []
    policies = []
    for route in ('reference', 'umi'):
        policy = FixturePolicy()
        policies.append(policy)
        with RunRecorder(tmp_path / route, metadata={}) as recorder:
            wrapped = bridge.RecordedPolicy(policy, route=route, recorder=recorder)
            results.append(wrapped.infer(request()))
    for key in request():
        np.testing.assert_equal(policies[0].requests[0][key], policies[1].requests[0][key])
    np.testing.assert_array_equal(results[0]['actions'], results[1]['actions'])
    assert results[1]['actions'].shape == (10, 7)
    np.testing.assert_array_equal(results[0]['state'], results[1]['state'])
    assert results[1]['policy_timing'] == results[0]['policy_timing']


def test_adapter_does_not_leak_targets_or_mutate_inputs():
    original = request()
    original['reward'] = 42
    original['target'] = np.ones(7)
    policy = FixturePolicy()
    adapter = LiberoChunkPolicy(policy)
    output = adapter(EnvironmentStep(observation=original, instruction=original['prompt']))
    assert set(policy.requests[0]) == set(request())
    policy.requests[0]['observation/image'][:] = 0
    np.testing.assert_array_equal(original['observation/image'], request()['observation/image'])
    output.action[:] = 0
    assert output.metadata['policy_output']['state'][1] == 1


@pytest.mark.parametrize('actions', [np.zeros(7), np.zeros((0, 7)), np.zeros((3, 8)),
                                    np.full((3, 7), np.nan), np.full((3, 7), np.inf)])
def test_invalid_actions_fail_without_becoming_a_benchmark_result(actions):
    class BadPolicy:
        def infer(self, observation):
            return {'actions': actions}
    with pytest.raises(ValueError, match='action chunk'):
        LiberoChunkPolicy(BadPolicy())(EnvironmentStep(observation=request(), instruction='task'))


@pytest.mark.parametrize('key,value', [('observation/state', np.zeros(7)),
                                     ('observation/state', np.full(8, np.nan)),
                                     ('observation/image', np.zeros((4, 4, 3)))])
def test_invalid_observations_do_not_reach_policy(key, value):
    observation = request()
    observation[key] = value
    policy = FixturePolicy()
    with pytest.raises(ValueError):
        LiberoChunkPolicy(policy)(EnvironmentStep(observation=observation, instruction='task'))
    assert not policy.requests


def test_recorded_replay_checks_every_call_and_detects_changed_actions(tmp_path):
    bridge, replay = example_module('serve_bridge'), example_module('replay')
    with RunRecorder(tmp_path / 'record', metadata={}) as recorder:
        policy = bridge.RecordedPolicy(FixturePolicy(), route='umi', recorder=recorder)
        for _ in range(3):
            policy.infer(request())
    assert replay.compare_record(tmp_path / 'record', FixturePolicy())['passed']

    class ChangedPolicy(FixturePolicy):
        def infer(self, observation):
            output = super().infer(observation)
            output['actions'][0, 0] += 0.25
            return output
    result = replay.compare_record(tmp_path / 'record', ChangedPolicy())
    assert result['failed_calls'] == 3 and not result['passed']
    assert result['max_absolute_error'] == 0.25


def test_policy_failure_marks_record_failed_and_replay_refuses_it(tmp_path):
    bridge, replay = example_module('serve_bridge'), example_module('replay')
    class FailedPolicy:
        def infer(self, observation):
            raise RuntimeError('inference failed')
    with RunRecorder(tmp_path / 'record', metadata={}) as recorder:
        policy = bridge.RecordedPolicy(FailedPolicy(), route='umi', recorder=recorder)
        with pytest.raises(RuntimeError, match='inference failed'):
            policy.infer(request())
    assert RunRecord(tmp_path / 'record').manifest['status'] == 'failed'
    with pytest.raises(ValueError, match='completed'):
        replay.compare_record(tmp_path / 'record', FixturePolicy())


def test_trial_audit_keeps_failed_trials_in_denominator(tmp_path):
    module = example_module('evaluate')
    audit = module.TrialAudit(tmp_path, horizon=5, num_steps_wait=10, max_steps=220)
    class Environment:
        def reset(self):
            pass
        def set_init_state(self, state):
            return state
        def step(self, action):
            return None, 0, bool(action), {}
    env = audit.wrap(Environment(), 'task')
    for success in (True, False):
        env.reset()
        env.set_init_state(np.zeros(3))
        env.step(success)
    audit.finish()
    assert len(audit.trials) == 2
    assert [t['success'] for t in audit.trials] == [True, False]
    assert [t['trial'] for t in audit.trials] == [0, 1]


def test_caught_upstream_error_remains_visible_in_trial_audit(tmp_path):
    module = example_module('evaluate')
    audit = module.TrialAudit(tmp_path, horizon=5, num_steps_wait=10, max_steps=220)
    audit.current = {'errors': []}
    handler = module.ErrorAudit(audit)
    handler.emit(logging.LogRecord('root', logging.ERROR, '', 0, 'inference failed', (), None))
    assert audit.errors == ['inference failed']
    assert audit.current['errors'] == audit.errors


def test_comparison_detects_pairing_errors_and_reports_both_disagreement_directions(tmp_path):
    module = example_module('compare')
    for route, outcomes in [('reference', [True, False, True]), ('umi', [False, True, True])]:
        directory = tmp_path / route
        directory.mkdir()
        report = {'status': 'complete', 'errors': [], 'expected_episodes': 3, 'successes': 2,
                  'suite': 'fixture', 'trials_per_task': 3, 'openpi_revision': 'test',
                  'libero_revision': 'test', 'evaluator_sha256': 'test', 'settings': {'seed': 7}}
        (directory / 'report.json').write_text(json.dumps(report))
        rows = [{'task': 'task', 'trial': i, 'success': success, 'errors': [],
                 'initial_state_sha256': str(i)} for i, success in enumerate(outcomes)]
        (directory / 'trials.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = module.compare(tmp_path / 'reference', tmp_path / 'umi')
    assert report['scope'] == {'reference': 'transport', 'umi': 'transport'}
    assert report['reference_only_successes'] == report['umi_only_successes'] == 1
    assert report['success_rate_difference'] == 0
    rows[0]['initial_state_sha256'] = 'wrong'
    (tmp_path / 'umi/trials.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='Initial state mismatch'):
        module.compare(tmp_path / 'reference', tmp_path / 'umi')


def test_chunk_metadata_does_not_change_external_scheduling():
    policy = FixturePolicy()
    response = LiberoChunkPolicy(policy, execution_horizon=5)(
        EnvironmentStep(observation=request(), instruction='task')
    )
    assert response.action.shape == (10, 7)
    assert response.metadata['prediction_horizon'] == 10
    assert response.metadata['execution_horizon'] == 5
    assert response.metadata['scheduler'] == 'external_evaluator'
    assert response.metadata['action_kind'] == 'chunk'
    unknown = LiberoChunkPolicy(policy)(EnvironmentStep(observation=request(), instruction='task'))
    assert unknown.metadata['execution_horizon'] is None


def test_chunk_must_cover_declared_execution_horizon():
    with pytest.raises(ValueError, match='execution_horizon'):
        LiberoChunkPolicy(FixturePolicy(), execution_horizon=11)(
            EnvironmentStep(observation=request(), instruction='task')
        )


# Completion protocol: ledger, receipt and comparison scope. Transport and
# end-to-end bridge checks are in test_libero_transport.py.

DUMMY = [0.0] * 6 + [-1.0]


class ChunkClient:
    """Stands in for BoundedClient; returns a distinct (10, 7) chunk per call."""

    def __init__(self):
        self.annotations = []

    def infer(self, observation, annotation):
        self.annotations.append(annotation)
        return {'actions': np.arange(70, dtype=np.float32).reshape(10, 7) + len(self.annotations)}


class StepEnvironment:
    def __init__(self, done_at=None, fail_at=None):
        self.done_at, self.fail_at, self.policy_steps = done_at, fail_at, 0

    def reset(self):
        self.policy_steps = 0

    def set_init_state(self, state):
        return state

    def step(self, action):
        if action == DUMMY:
            return None, 0, False, {}
        self.policy_steps += 1
        if self.policy_steps == self.fail_at:
            raise RuntimeError('simulator failed')
        return None, 0, self.policy_steps == self.done_at, {}


def run_episode(audit, env, horizon, max_steps, wait=2):
    """The upstream loop's chunk scheduling, as in eval_libero, for one episode."""
    policy = audit.wrap_policy(ChunkClient())
    env.reset()
    env.set_init_state(np.zeros(3))
    plan = []
    for t in range(max_steps + wait):
        try:
            if t < wait:
                env.step(DUMMY)
                continue
            if not plan:
                plan.extend(policy.infer({})['actions'][:horizon])
            if env.step(plan.pop(0).tolist())[2]:
                break
        except Exception as error:
            logging.getLogger().error('Caught exception: %s', error)
            break
    audit.finish()
    return audit.trials[-1]


@pytest.mark.parametrize('done_at,max_steps,ended_by,calls,last,full', [
    (4, 10, 'success', 2, 1, 1),        # success inside a chunk
    (6, 10, 'success', 2, 3, 2),        # success on a chunk boundary
    (None, 10, 'step_limit', 4, 1, 3),  # step limit inside a chunk
    (None, 9, 'step_limit', 3, 3, 3),   # step limit on a chunk boundary
])
def test_ledger_counts_whole_chunks_and_partial_last_chunk(tmp_path, done_at, max_steps, ended_by,
                                                          calls, last, full):
    evaluate, bridge = example_module('evaluate'), example_module('serve_tool_bridge')
    audit = evaluate.TrialAudit(tmp_path, horizon=3, num_steps_wait=2, max_steps=max_steps)
    row = run_episode(audit, audit.wrap(StepEnvironment(done_at), 'task'), 3, max_steps)
    assert (row['ended_by'], row['calls'], row['last_executed'], row['full_calls']) == (
        ended_by, calls, last, full)
    assert row['executed_total'] == 3 * (calls - 1) + last and row['wait_steps'] == 2
    assert row['prefix_match'] and not row['execution_unknown']
    # The bridge accepts the same episode independently.
    episode = dict({k: row[k] for k in evaluate.LEDGER_FIELDS}, errors=0)
    start = {'num_steps_wait': 2, 'max_steps': max_steps}
    bridge.check_receipt(receipt([episode]), run_id='run', start=start, keys=[('task', 0)],
                         calls=[calls], horizon=3)


def test_ledger_marks_unexplained_stop_raised_step_and_changed_action_as_errors(tmp_path):
    evaluate = example_module('evaluate')
    audit = evaluate.TrialAudit(tmp_path, horizon=3, num_steps_wait=2, max_steps=10)
    handler = evaluate.ErrorAudit(audit)
    logging.getLogger().addHandler(handler)
    try:
        raised = run_episode(audit, audit.wrap(StepEnvironment(fail_at=2), 'task'), 3, 10)
    finally:
        logging.getLogger().removeHandler(handler)
    assert raised['execution_unknown'] and raised['executed_total'] == 1
    assert raised['ended_by'] == 'error' and raised['errors']
    # A stop short of the step limit with no done and no logged error.
    stopped = run_episode(audit, audit.wrap(StepEnvironment(), 'task'), 3, 5)
    assert stopped['ended_by'] == 'error' and not stopped['errors']
    # An action that is not the next action of the received chunk.
    env = audit.wrap(StepEnvironment(done_at=1), 'task')
    env.reset()
    for _ in range(2):
        env.step(DUMMY)
    audit.wrap_policy(ChunkClient()).infer({})
    env.step([9.0] * 7)
    audit.finish()
    assert not audit.trials[-1]['prefix_match'] and audit.trials[-1]['ended_by'] == 'error'
    assert [t['trial'] for t in audit.trials] == [0, 1, 2]


def start_message(**changes):
    start = {'version': 'brainscore.libero.receipt.v1', 'run_id': 'run', 'suite': 'libero_spatial',
             'trials_per_task': 2, 'execution_horizon': 5, 'num_steps_wait': 10,
             'max_steps': 220, 'tasks': ['a', 'b']}
    start.update(changes)
    return start


@pytest.mark.parametrize('changes', [
    {'version': 'v0'}, {'run_id': 'other'}, {'suite': 'libero_goal'}, {'trials_per_task': 3},
    {'trials_per_task': True}, {'execution_horizon': 4}, {'max_steps': 0}, {'max_steps': 2.5},
    {'num_steps_wait': -1}, {'tasks': []}, {'tasks': ['a', 'a']}, {'tasks': ['a', '']},
    {'tasks': ['t%d' % i for i in range(91)]}, {'extra': 1}, {'tasks': [['a']]}, {'tasks': 'ab'},
])
def test_start_message_must_match_bridge_configuration(changes):
    bridge = example_module('serve_tool_bridge')
    settings = dict(run_id='run', suite='libero_spatial', trials_per_task=2, horizon=5)
    assert bridge.check_start(start_message(), **settings) == [('a', 0), ('a', 1), ('b', 0), ('b', 1)]
    with pytest.raises(ValueError):
        bridge.check_start(start_message(**changes), **settings)


def receipt(episodes, **changes):
    value = {'version': 'brainscore.libero.receipt.v1', 'run_id': 'run', 'report_sha256': 'a' * 64,
             'status': 'complete', 'episodes': episodes}
    value.update(changes)
    return value


def good_episodes():
    common = dict(full_calls=44, last_executed=5, executed_total=220, wait_steps=10, errors=0,
                  prefix_match=True, execution_unknown=False)
    return [dict(common, task='a', trial=0, calls=44, success=False, ended_by='step_limit'),
            dict(common, task='a', trial=1, calls=3, full_calls=2, last_executed=2,
                 executed_total=12, success=True, ended_by='success')]


@pytest.mark.parametrize('episode,changes', [
    (None, {'status': 'failed'}), (None, {'run_id': 'other'}), (None, {'report_sha256': 'A' * 64}),
    (None, {'version': 'v0'}), (None, {'episodes': []}),
    (1, {'calls': 4}), (1, {'errors': 1}), (1, {'prefix_match': False}),
    (1, {'execution_unknown': True}), (1, {'ended_by': 'error'}), (1, {'ended_by': 'step_limit'}),
    (1, {'success': 1}), (1, {'wait_steps': 9}), (1, {'last_executed': 0, 'executed_total': 10}),
    (1, {'full_calls': 3}), (1, {'executed_total': 13}), (1, {'trial': 0}), (1, {'calls': True}),
    (1, {'trial': True}), (0, {'trial': False}), (0, {'task': ['a']}),
    (0, {'success': True, 'ended_by': 'success', 'executed_total': 221, 'last_executed': 6}),
    (0, {'executed_total': 219, 'last_executed': 4, 'full_calls': 43}), (1, {'note': 'x'}),
])
def test_receipt_is_rejected_unless_every_episode_is_consistent(episode, changes):
    bridge = example_module('serve_tool_bridge')
    settings = dict(run_id='run', start=start_message(), keys=[('a', 0), ('a', 1)], calls=[44, 3],
                    horizon=5)
    bridge.check_receipt(receipt(good_episodes()), **settings)
    episodes = good_episodes()
    if episode is None:
        value = dict(receipt(episodes), **changes)
    else:
        episodes[episode].update(changes)
        value = receipt(episodes)
    with pytest.raises(ValueError):
        bridge.check_receipt(value, **settings)


class FakeUpstream:
    """Enough of the pinned evaluator module to test finalization without LIBERO."""

    def __init__(self, evaluate, close_fails=True):
        self.close_fails = close_fails
        from dataclasses import dataclass
        from types import SimpleNamespace

        @dataclass
        class Args:
            host: str = '0.0.0.0'
            port: int = 8000
            replan_steps: int = 5
            task_suite_name: str = 'libero_spatial'
            num_steps_wait: int = 10
            num_trials_per_task: int = 50
            video_out_path: str = 'videos'
            seed: int = 7

        suite = SimpleNamespace(n_tasks=1, get_task=lambda i: SimpleNamespace(name='task'))
        self.Args = Args
        self.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {'libero_spatial': lambda: suite})
        self.imageio = SimpleNamespace(mimwrite=lambda *args, **kwargs: None)
        self.closed = []
        self.evaluate = evaluate

    def _get_libero_env(self, task, resolution, seed):
        upstream = self

        class Environment(StepEnvironment):
            def close(self):
                upstream.closed.append(True)
                if upstream.close_fails:
                    raise OSError('close failed')
        return Environment(done_at=1), 'task'


def evaluator_args(tmp_path, **changes):
    from argparse import Namespace
    values = dict(out=tmp_path / 'out', route='umi', suite='libero_spatial', trials=1, port=1,
                  connect_timeout=1.0, infer_timeout=1.0, control_timeout=1.0, finish_timeout=1.0)
    values.update(changes)
    return Namespace(**values)


def test_cleanup_failure_still_reports_failure_and_original_error_propagates(tmp_path):
    evaluate = example_module('evaluate')
    upstream = FakeUpstream(evaluate)

    def eval_libero(config):
        from types import SimpleNamespace
        env, _ = upstream._get_libero_env(SimpleNamespace(name='task'), 256, 7)
        env.reset()
        raise RuntimeError('evaluator stopped')
    upstream.eval_libero = eval_libero
    source = tmp_path / 'main.py'
    source.write_text('# pinned evaluator stand-in\n')
    with pytest.raises(RuntimeError, match='evaluator stopped'):
        evaluate.run(upstream, source, evaluator_args(tmp_path))
    report = json.loads((tmp_path / 'out/report.json').read_text())
    assert report['status'] == 'failed' and upstream.closed
    assert report['cleanup_errors'] == ["environment: OSError('close failed')"]
    assert report['bridge'] is None and not (tmp_path / 'out/finish.json').exists()


def test_unsaved_report_is_a_failure_without_claiming_an_artifact(tmp_path, monkeypatch):
    evaluate = example_module('evaluate')
    upstream = FakeUpstream(evaluate)
    upstream.eval_libero = lambda config: None
    source = tmp_path / 'main.py'
    source.write_text('# pinned evaluator stand-in\n')
    write_bytes = Path.write_bytes

    def full_disk(path, data):
        if path.name == 'report.json':
            raise OSError('No space left on device')
        return write_bytes(path, data)
    monkeypatch.setattr(Path, 'write_bytes', full_disk)
    with pytest.raises(SystemExit):
        evaluate.run(upstream, source, evaluator_args(tmp_path))
    assert not (tmp_path / 'out/report.json').exists()


def write_route(directory, *, bridge=None):
    directory.mkdir()
    report = {'status': 'complete', 'errors': [], 'expected_episodes': 1, 'successes': 1,
              'suite': 'fixture', 'trials_per_task': 1, 'openpi_revision': 'test',
              'libero_revision': 'test', 'evaluator_sha256': 'test', 'settings': {'seed': 7},
              'bridge': bridge}
    (directory / 'report.json').write_text(json.dumps(report))
    row = {'task': 'task', 'trial': 0, 'success': True, 'errors': [], 'initial_state_sha256': '0'}
    (directory / 'trials.jsonl').write_text(json.dumps(row) + '\n')


def test_comparison_scope_requires_experiment_records_only_for_tool_bridge_reports(tmp_path):
    module = example_module('compare')
    write_route(tmp_path / 'reference')
    write_route(tmp_path / 'tool', bridge={'receipt_protocol': 'brainscore.libero.receipt.v1',
                                           'umi_run_id': 'run', 'experiment_tools': 'baseline'})
    with pytest.raises(ValueError, match='pass its UMI experiment'):
        module.compare(tmp_path / 'reference', tmp_path / 'tool')
    with pytest.raises(ValueError, match='transport-only'):
        module.compare(tmp_path / 'reference', tmp_path / 'reference', umi_experiment=tmp_path)


@pytest.mark.parametrize('name', ['evaluate', 'compare'])
def test_evaluator_side_scripts_parse_as_python_38(name):
    import ast
    path = Path(__file__).parents[1] / 'examples' / 'libero' / (name + '.py')
    ast.parse(path.read_text(), feature_version=(3, 8))


@pytest.mark.parametrize('changes', [
    {'mode': 'other'}, {'suite': 'libero_90'}, {'trials_per_task': True}, {'trials_per_task': 0},
    {'trials_per_task': 51}, {'horizon': True}, {'horizon': 0}, {'horizon': 1.5},
    {'idle_timeout': True}, {'idle_timeout': 0}, {'idle_timeout': float('inf')},
    {'idle_timeout': float('nan')}, {'idle_timeout': '900'},
])
def test_bridge_rejects_invalid_configuration_before_connecting(tmp_path, changes):
    bridge = example_module('serve_tool_bridge')
    settings = dict(upstream='ws://127.0.0.1:1', output=tmp_path, mode='baseline',
                    suite='libero_spatial', trials_per_task=2, horizon=5, idle_timeout=900)
    bridge.Bridge(**settings)
    with pytest.raises(ValueError):
        bridge.Bridge(**dict(settings, **changes))


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan'), True, '5'])
def test_bounded_client_rejects_invalid_timeouts_before_connecting(timeout):
    evaluate = example_module('evaluate')
    for timeouts in ({'connect_timeout': timeout, 'infer_timeout': 1},
                     {'connect_timeout': 1, 'infer_timeout': timeout}):
        with pytest.raises(ValueError, match='Timeouts'):
            evaluate.BoundedClient('ws://127.0.0.1:1', **timeouts)


class AckClient:
    """A receipt-mode BoundedClient whose bridge accepts everything."""
    receipt, run_id = True, 'run'
    bridge = {'receipt_protocol': 'brainscore.libero.receipt.v1', 'umi_run_id': 'run',
              'experiment_tools': 'baseline'}
    receipts = []

    def __init__(self, uri, *, connect_timeout, infer_timeout):
        self.broken = False

    def infer(self, observation, annotation):
        return {'actions': np.zeros((10, 7), dtype=np.float32)}

    def control(self, message, timeout):
        if 'umi/start' in message:
            return {'umi/start_ack': 'accepted', 'run_id': 'run'}
        AckClient.receipts.append(message['umi/receipt'])
        return {'umi/finish_ack': 'complete', 'run_id': 'run', 'manifest_sha256': 'b' * 64}

    def close(self):
        self.broken = True


def one_successful_episode(upstream, log_error=False):
    def eval_libero(config):
        from types import SimpleNamespace
        policy = upstream._websocket_client_policy.WebsocketClientPolicy(config.host, config.port)
        env, _ = upstream._get_libero_env(SimpleNamespace(name='task'), 256, 7)
        env.reset()
        env.set_init_state(np.zeros(3))
        for _ in range(config.num_steps_wait):
            env.step(DUMMY)
        env.step(policy.infer({})['actions'][0].tolist())
        if log_error:
            logging.getLogger().error('Caught exception: swallowed by the evaluator')
    return eval_libero


@pytest.mark.parametrize('fault', [None, 'finish.json', 'swallowed'])
def test_unsaved_acknowledgment_or_swallowed_error_never_reports_success(tmp_path, monkeypatch, fault):
    evaluate = example_module('evaluate')
    monkeypatch.setattr(evaluate, 'BoundedClient', AckClient)
    AckClient.receipts.clear()
    upstream = FakeUpstream(evaluate, close_fails=False)
    upstream.eval_libero = one_successful_episode(upstream, log_error=fault == 'swallowed')
    source = tmp_path / 'main.py'
    source.write_text('# pinned evaluator stand-in\n')
    write_text = Path.write_text

    def failing(path, data, *args, **kwargs):
        if path.name == fault:
            raise OSError('No space left on device')
        return write_text(path, data, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', failing)
    if fault is None:
        evaluate.run(upstream, source, evaluator_args(tmp_path))  # positive control: exits normally
    else:
        with pytest.raises(SystemExit):
            evaluate.run(upstream, source, evaluator_args(tmp_path))
    report = json.loads((tmp_path / 'out/report.json').read_text())
    assert report['status'] == ('failed' if fault == 'swallowed' else 'complete')
    assert AckClient.receipts[-1]['status'] == report['status']
    assert (tmp_path / 'out/finish.json').exists() == (fault != 'finish.json')


def test_original_evaluator_error_survives_failed_cleanup_and_unsaved_report(tmp_path, monkeypatch):
    evaluate = example_module('evaluate')
    upstream = FakeUpstream(evaluate)

    def eval_libero(config):
        from types import SimpleNamespace
        upstream._get_libero_env(SimpleNamespace(name='task'), 256, 7)[0].reset()
        raise KeyError('original evaluator error')
    upstream.eval_libero = eval_libero
    source = tmp_path / 'main.py'
    source.write_text('# pinned evaluator stand-in\n')
    monkeypatch.setattr(Path, 'write_bytes', lambda path, data: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(KeyError, match='original evaluator error'):
        evaluate.run(upstream, source, evaluator_args(tmp_path))
    assert upstream.closed and not (tmp_path / 'out/report.json').exists()
