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
    audit = module.TrialAudit(tmp_path)
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
    audit = module.TrialAudit(tmp_path)
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
