"""Offline contract checks; these are not trained-policy qualification."""
from contextlib import contextmanager
import importlib.util
import json
import logging
import os
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


def capture(directory, calls, *, route='reference'):
    """A transport capture as serve_bridge.py writes it, with completed calls only."""
    bridge = example_module('serve_bridge')
    with RunRecorder(directory, metadata=bridge.capture_metadata(route, {})) as recorder:
        policy = bridge.RecordedPolicy(FixturePolicy(), route=route, recorder=recorder)
        for _ in range(calls):
            policy.infer(request())
    return directory


def evaluator_report(directory, calls=(2, 1), *, route='reference', rows=None, **changes):
    """A completed evaluate.py report: one trial per task, each ending in success."""
    evaluate = example_module('evaluate')
    rows = rows if rows is not None else [
        {'task': f'task-{i}', 'trial': 0, 'steps': 10 + 5 * (n - 1) + 2, 'success': True,
         'errors': [], 'calls': n, 'full_calls': n - 1, 'last_executed': 2,
         'executed_total': 5 * (n - 1) + 2, 'wait_steps': 10, 'prefix_match': True,
         'execution_unknown': False, 'ended_by': 'success', 'initial_state_sha256': str(i)}
        for i, n in enumerate(calls)]
    report = {'status': 'complete', 'route': route, 'suite': 'libero_spatial',
              'trials_per_task': 1, 'episodes': len(rows), 'expected_episodes': len(rows),
              'successes': sum(row['success'] for row in rows), 'success_rate': 1.0,
              'errors': [], 'cleanup_errors': [], 'execution_horizon': 5, 'num_steps_wait': 10,
              'max_steps': 220, 'bridge': None, 'openpi_revision': evaluate.OPENPI_REVISION,
              'libero_revision': evaluate.LIBERO_REVISION, 'evaluator_sha256': '0' * 64,
              'settings': {'task_suite_name': 'libero_spatial', 'num_trials_per_task': 1,
                           'replan_steps': 5, 'num_steps_wait': 10, 'seed': 7}}
    report.update(changes)
    directory.mkdir()
    (directory / 'report.json').write_text(json.dumps(report))
    (directory / 'trials.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return directory


def test_recorded_replay_checks_every_call_and_detects_changed_actions(tmp_path):
    replay = example_module('replay')
    record, report = capture(tmp_path / 'record', 3), evaluator_report(tmp_path / 'eval')
    result = replay.compare_record(record, FixturePolicy(), report=report,
                                   suite='libero_spatial', trials_per_task=1)
    assert result['passed'] and result['calls'] == result['source']['calls'] == 3

    class ChangedPolicy(FixturePolicy):
        def infer(self, observation):
            output = super().infer(observation)
            output['actions'][0, 0] += 0.25
            return output
    result = replay.compare_record(record, ChangedPolicy(), report=report)
    assert result['failed_calls'] == 3 and not result['passed']
    assert result['max_absolute_error'] == 0.25


def test_policy_failure_marks_record_failed_and_replay_refuses_it(tmp_path):
    bridge, replay = example_module('serve_bridge'), example_module('replay')
    class FailedPolicy:
        def infer(self, observation):
            raise RuntimeError('inference failed')
    metadata = bridge.capture_metadata('reference', {})
    with RunRecorder(tmp_path / 'record', metadata=metadata) as recorder:
        policy = bridge.RecordedPolicy(FailedPolicy(), route='reference', recorder=recorder)
        with pytest.raises(RuntimeError, match='inference failed'):
            policy.infer(request())
    assert RunRecord(tmp_path / 'record').manifest['status'] == 'failed'
    with pytest.raises(ValueError, match='completed'):
        replay.compare_record(tmp_path / 'record', FixturePolicy(),
                              report=evaluator_report(tmp_path / 'eval', (1,)))


def test_interrupted_call_fails_the_capture_immediately(tmp_path):
    # The bridge's own stop handler raises KeyboardInterrupt inside the blocked call.
    bridge = example_module('serve_bridge')
    class Interrupted:
        def infer(self, observation):
            raise KeyboardInterrupt
    recorder = RunRecorder(tmp_path / 'record', metadata=bridge.capture_metadata('reference', {}))
    policy = bridge.RecordedPolicy(Interrupted(), route='reference', recorder=recorder)
    with pytest.raises(KeyboardInterrupt):
        policy.infer(request())
    manifest = RunRecord(tmp_path / 'record').manifest
    assert manifest['status'] == 'failed' and manifest['event_count'] == 1  # input, no reply
    recorder.close()  # a later clean close cannot turn it into a success
    assert RunRecord(tmp_path / 'record').manifest['status'] == 'failed'


def _partial_chunks(rows):
    rows[1]['executed_total'] = 1  # counts that no whole-chunk schedule can produce
    return rows


def _missing_field(rows):
    return [{k: v for k, v in row.items() if k != 'wait_steps'} for row in rows]


@pytest.mark.parametrize('change,message', [
    ({'status': 'failed'}, 'completed transport'),
    ({'route': 'umi'}, 'completed transport'),
    ({'bridge': {'receipt_protocol': 'x'}}, 'completed transport'),
    ({'errors': ['inference failed']}, 'reported errors'),
    ({'cleanup_errors': ['environment: x']}, 'reported errors'),
    ({'openpi_revision': '0' * 40}, 'pinned evaluator'),
    ({'max_steps': 300}, 'inconsistent'),
    ({'settings': {'task_suite_name': 'libero_spatial', 'num_trials_per_task': 1,
                   'replan_steps': 4, 'num_steps_wait': 10}}, 'inconsistent'),
    ({'expected_episodes': 3}, 'totals'),
    ({'successes': 1}, 'totals'),
    ({'calls': (2, 2)}, 'accounts for 4 calls'),
    ({'calls': (1, 1)}, 'accounts for 2 calls'),
    ({'mutate': _partial_chunks}, 'whole chunks'),
    ({'mutate': _missing_field}, 'Malformed'),
])
def test_capture_needs_a_matching_completed_evaluator_report(tmp_path, change, message):
    replay = example_module('replay')
    record = capture(tmp_path / 'record', 3)
    calls, mutate = change.pop('calls', (2, 1)), change.pop('mutate', None)
    rows = None
    if mutate is not None:
        good = evaluator_report(tmp_path / 'good', calls)
        rows = mutate([json.loads(line) for line in (good / 'trials.jsonl').read_text().splitlines()])
    report = evaluator_report(tmp_path / 'eval', calls, rows=rows, **change)
    with pytest.raises(ValueError, match=message):
        replay.compare_record(record, FixturePolicy(), report=report)


def test_capture_refuses_requested_settings_missing_reports_and_old_or_unpaired_records(tmp_path):
    bridge, replay = example_module('serve_bridge'), example_module('replay')
    record, report = capture(tmp_path / 'record', 3), evaluator_report(tmp_path / 'eval')
    with pytest.raises(ValueError, match='suite or trials'):
        replay.compare_record(record, FixturePolicy(), report=report, suite='libero_goal')
    with pytest.raises(ValueError, match='suite or trials'):
        replay.compare_record(record, FixturePolicy(), report=report, trials_per_task=50)
    with pytest.raises(FileNotFoundError):
        replay.compare_record(record, FixturePolicy(), report=tmp_path / 'absent')
    with pytest.raises(TypeError):
        replay.compare_record(record, FixturePolicy())  # the report is required
    # A capture from the bridge before this check could not detect interrupted calls.
    with RunRecorder(tmp_path / 'old', metadata={'route': 'reference'}) as recorder:
        policy = bridge.RecordedPolicy(FixturePolicy(), route='reference', recorder=recorder)
        for _ in range(3):
            policy.infer(request())
    with pytest.raises(ValueError, match='older captures'):
        replay.compare_record(tmp_path / 'old', FixturePolicy(), report=report)
    # Right event count and status, but not request/reply pairs.
    with RunRecorder(tmp_path / 'unpaired', metadata=bridge.capture_metadata('reference', {})) as recorder:
        for direction in ('input', 'input', 'output', 'output', 'input', 'output'):
            recorder.record(direction, request())
    with pytest.raises(ValueError, match='alternating'):
        replay.compare_record(tmp_path / 'unpaired', FixturePolicy(), report=report)


def test_malformed_pair_beyond_a_selected_prefix_is_refused(tmp_path):
    # qualify_*.py take only the first requests; verify_capture must still read every pair.
    bridge, replay = example_module('serve_bridge'), example_module('replay')
    metadata = bridge.capture_metadata('reference', {})
    with RunRecorder(tmp_path / 'record', metadata=metadata) as recorder:
        for direction in ('input', 'output', 'input', 'output', 'output', 'input'):
            recorder.record(direction, request())
    with pytest.raises(ValueError, match='alternating'):
        replay.verify_capture(RunRecord(tmp_path / 'record'), evaluator_report(tmp_path / 'eval'))


class UnitPolicy:
    """Hidden unit 0 is recorded and can be ablated; actions use only unit 1."""
    _rng = 0

    def __init__(self, ablation_works=True):
        self.scopes, self.ablation_works = [], ablation_works

    def infer(self, observation, **kwargs):
        hidden = np.array([1 + observation['observation/state'][0], 2], dtype=np.float32)
        for operation, receive in self.scopes:
            if operation == 'ablate' and self.ablation_works:
                hidden[0] = 0
            elif operation == 'record':
                receive('action_out_proj', {'array': hidden[:1].copy(), 'denoising_step': 0})
        return {'actions': np.full((10, 7), hidden[1], dtype=np.float32)}


class UnitInstrumentation:
    """Stands in for OpenPIInstrumentation; scopes apply in attachment order."""

    def __init__(self, policy, *, checkpoint, steps=None):
        self.policy, self.checkpoint = policy, checkpoint

    def describe(self):
        return {'provider': 'unit-fixture', 'checkpoint': self.checkpoint}

    def validate(self, operation, targets):
        if operation not in ('record', 'ablate'):
            raise ValueError('Unsupported operation')

    @contextmanager
    def _scope(self, operation, receive=None):
        self.policy.scopes.append((operation, receive))
        try:
            yield
        finally:
            self.policy.scopes.pop()

    def record(self, targets, receive):
        return self._scope('record', receive)

    def ablate(self, targets):
        return self._scope('ablate')


@pytest.mark.parametrize('ablation_works', [True, False])
def test_unchanged_actions_after_ablation_are_reported_not_gated(tmp_path, monkeypatch, ablation_works):
    import brainscore.experiments as experiments
    monkeypatch.setattr(experiments, 'OpenPIInstrumentation', UnitInstrumentation)
    report = example_module('qualify_tools').qualify(
        UnitPolicy(ablation_works), [request(), request()], tmp_path / 'q',
        checkpoint='unit-fixture', steps=[0])
    assert report['effects'] == {'intervention_changed_actions': False}
    assert report['recording_preserves_actions'] and report['cleanup_restores_actions']
    assert report['selected_activity_zeroed'] is ablation_works
    assert report['status'] == ('passed' if ablation_works else 'failed')
    assert json.loads((tmp_path / 'q/qualification.json').read_text())['status'] == report['status']


RUN_TOOLS = Path(__file__).parents[1] / 'examples' / 'libero' / 'run_tools.sh'


def run_tools(tmp_path, *arguments, **environment):
    import subprocess
    import sys
    (tmp_path / 'openpi').mkdir(exist_ok=True)
    (tmp_path / 'checkpoint').mkdir(exist_ok=True)
    env = {'PATH': os.environ['PATH'], 'UMI_POLICY_PYTHON': sys.executable,
           'UMI_BRIDGE_PYTHON': sys.executable, 'UMI_EVALUATOR_PYTHON': sys.executable,
           'UMI_POLICY_PORT': str(free_port()), 'UMI_BRIDGE_PORT': str(free_port())}
    env.update(environment)
    command = ['bash', str(RUN_TOOLS), str(tmp_path / 'openpi'), str(tmp_path / 'checkpoint'),
               str(tmp_path / 'evidence'), *arguments]
    return subprocess.run(command, env={k: v for k, v in env.items() if v is not None},
                          capture_output=True, text=True, timeout=120)


def free_port():
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def test_tool_route_driver_parses():
    import subprocess
    subprocess.run(['bash', '-n', str(RUN_TOOLS)], check=True)


@pytest.mark.parametrize('arguments,environment,message', [
    ((), {}, 'Usage'),
    (('libero_spatial', '1', '60'), {'UMI_BRIDGE_PYTHON': None}, 'UMI_BRIDGE_PYTHON'),
    (('libero_90', '1', '60'), {}, 'Unknown suite'),
    (('libero_spatial', '51', '60'), {}, '1 to 50'),
    (('libero_spatial', '1', '0'), {}, 'CONDITION_SECONDS'),
    (('libero_spatial', '1', '60', 'baseline'), {}, 'Unknown mode'),
    (('libero_spatial', '1', '60'), {'UMI_READY_SECONDS': 'soon'}, 'UMI_READY_SECONDS'),
    (('libero_spatial', '1', '60', 'recording', 'recording'), {}, 'each mode only once'),
    (('libero_spatial', '1', '60'), {'UMI_POLICY_PORT': '8000', 'UMI_BRIDGE_PORT': '8000'}, 'distinct ports'),
    (('libero_spatial', '1', '60'), {'UMI_POLICY_PORT': '0'}, 'distinct ports'),
    (('libero_spatial', '1', '60'), {'UMI_BRIDGE_PORT': 'invalid'}, 'distinct ports'),
])
def test_tool_route_driver_rejects_invalid_settings_before_starting_anything(
        tmp_path, arguments, environment, message):
    result = run_tools(tmp_path, *arguments, **environment)
    assert result.returncode == 2 and message in result.stderr
    assert not (tmp_path / 'evidence').exists()


def test_tool_route_driver_never_reuses_evidence(tmp_path):
    (tmp_path / 'evidence').mkdir()
    (tmp_path / 'evidence' / 'keep').write_text('earlier run')
    result = run_tools(tmp_path, 'libero_spatial', '1', '60')
    assert result.returncode != 0
    assert [p.name for p in (tmp_path / 'evidence').iterdir()] == ['keep']


def test_tool_route_driver_stops_when_the_policy_server_exits_before_ready(tmp_path):
    # A policy "interpreter" that exits at once: no bridge or evaluator may start.
    result = run_tools(tmp_path, 'libero_spatial', '1', '60', UMI_POLICY_PYTHON='false')
    assert result.returncode != 0 and 'exited before it was ready' in result.stderr
    evidence = tmp_path / 'evidence'
    assert (evidence / 'baseline-policy.log').exists() and (evidence / 'revisions.txt').exists()
    assert not (evidence / 'baseline-bridge.log').exists()
    assert not (evidence / 'baseline-eval').exists()


def test_qualification_gates_survive_python_optimization():
    import ast
    import subprocess
    import sys
    root = Path(__file__).parents[1] / 'examples' / 'libero'
    for name in ('serve_bridge', 'replay', 'toolbox_calls', 'qualify_tools', 'qualify_remote_tools'):
        tree = ast.parse((root / (name + '.py')).read_text())
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)], name
    script = ('import importlib.util, sys; spec = importlib.util.spec_from_file_location('
              '"q", sys.argv[1]); m = importlib.util.module_from_spec(spec); '
              'spec.loader.exec_module(m); m._check(False, "gate ran")')
    result = subprocess.run([sys.executable, '-O', '-c', script, str(root / 'qualify_remote_tools.py')],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'gate ran' in result.stderr


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


def test_evaluator_deadline_helper_parses_as_python_38_and_loads_without_brainscore():
    # Syntax only: no Python 3.8 interpreter runs here.
    import ast
    import subprocess
    import sys
    root = Path(__file__).parents[1]
    helper = root / 'brainscore/experiments/backends/socket_deadline.py'
    ast.parse(helper.read_text(), feature_version=(3, 8))
    check = ("import importlib.util, sys\n"
             "spec = importlib.util.spec_from_file_location('evaluate', sys.argv[1])\n"
             "module = importlib.util.module_from_spec(spec)\n"
             "spec.loader.exec_module(module)\n"
             "assert module.socket_deadline.__file__ == sys.argv[2]\n"
             "assert not [m for m in sys.modules if m.split('.')[0] == 'brainscore'], 'imported brainscore'\n")
    subprocess.run([sys.executable, '-c', check, str(root / 'examples/libero/evaluate.py'),
                    str(helper.resolve())], check=True, cwd=str(root / 'examples'))


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
