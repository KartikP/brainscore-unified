"""Opt-in local websocket tests using pinned OpenPI code and untrained fixtures.

Set OPENPI_SOURCE to a checkout; no checkpoint, simulator, or GPU is used.
The evaluator tests run the pinned eval_libero source around a fake simulator.
They check transport and completion only, not simulator or policy behavior.
"""
from argparse import Namespace
import ast
import collections
from contextlib import contextmanager
import dataclasses
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import types
from types import SimpleNamespace
import urllib.request

import numpy as np
import pytest

if not os.environ.get('OPENPI_SOURCE'):
    pytest.skip('Requires pinned OpenPI source', allow_module_level=True)
pytestmark = pytest.mark.integration
ROOT = Path(__file__).parents[1]
OPENPI = Path(os.environ['OPENPI_SOURCE']).resolve()


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@contextmanager
def server(command, port, env, log):
    with log.open('w') as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 20
            while True:
                if process.poll() is not None:
                    raise RuntimeError(log.read_text())
                try:
                    urllib.request.urlopen('http://127.0.0.1:%d/healthz' % port, timeout=1).close()
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(log.read_text())
                    time.sleep(0.1)
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def test_reference_and_umi_over_real_websocket_with_record_replay(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    openpi = Path(os.environ['OPENPI_SOURCE']).resolve()
    for path in (openpi / 'src', openpi / 'packages/openpi-client/src'):
        monkeypatch.syspath_prepend(str(path))
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(openpi / 'src'), str(openpi / 'packages/openpi-client/src'),
                                       *sys.path])
    policy_port, bridge_port = free_port(), free_port()
    fixture = '''
import numpy as np
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
class Policy:
    def infer(self, request):
        value = request['observation/state'][0]
        return {'actions': np.arange(21, dtype=np.float32).reshape(3,7) + value}
WebsocketPolicyServer(Policy(), host='127.0.0.1', port=%d).serve_forever()
''' % policy_port
    requests = [{'observation/image': np.full((4, 4, 3), i, dtype=np.uint8),
                 'observation/wrist_image': np.zeros((4, 4, 3), dtype=np.uint8),
                 'observation/state': np.full(8, i, dtype=np.float32), 'prompt': 'task'}
                for i in range(3)]
    with server([sys.executable, '-c', fixture], policy_port, env, tmp_path / 'policy.log'):
        for route in ('reference', 'umi'):
            command = [sys.executable, str(root / 'examples/libero/serve_bridge.py'), '--route', route,
                       '--upstream-port', str(policy_port), '--port', str(bridge_port),
                       '--out', str(tmp_path / route)]
            with server(command, bridge_port, env, tmp_path / (route + '.log')):
                client = WebsocketClientPolicy('127.0.0.1', bridge_port)
                try:
                    for i, request in enumerate(requests):
                        actual = client.infer(request)['actions']
                        np.testing.assert_array_equal(actual, np.arange(21).reshape(3, 7) + i)
                    if route == 'umi':
                        path = root / 'examples/libero/replay.py'
                        spec = importlib.util.spec_from_file_location('libero_replay_transport', path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        assert module.compare_record(tmp_path / 'reference', client)['passed']
                finally:
                    client._ws.close()
        # The same external policy is also usable through the common toolbox runner.
        path = root / 'examples/libero/toolbox_calls.py'
        spec = importlib.util.spec_from_file_location('toolbox_calls_transport', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        client = WebsocketClientPolicy('127.0.0.1', policy_port)
        try:
            result = module.run_calls(client, tmp_path / 'reference', tmp_path / 'toolbox')
            assert result.value['calls'] == 3 and result.value['exact']
            assert result.manifest['status'] == 'complete'
        finally:
            client._ws.close()
    from brainscore.run_record import RunRecord
    assert RunRecord(tmp_path / 'reference').manifest['status'] == 'complete'
    assert RunRecord(tmp_path / 'umi').manifest['status'] == 'complete'


# Completion protocol between evaluate.py and serve_tool_bridge.py.

def example(name):
    path = ROOT / 'examples' / 'libero' / (name + '.py')
    spec = importlib.util.spec_from_file_location('libero_completion_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUIET = logging.getLogger('test-libero-completion')
QUIET.propagate = False  # keep server-side errors out of the evaluator's ErrorAudit
TASKS = {'a': 7, 'b': 10, 'c': None}  # done after this many policy steps; None: step limit


@contextmanager
def listening(handler):
    from websockets.sync.server import serve
    with serve(handler, '127.0.0.1', 0, compression=None, max_size=64 * 1024 * 1024,
               logger=QUIET) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.socket.getsockname()[1]
        finally:
            server.shutdown()
            thread.join(timeout=10)


class ChunkPolicy:
    def __init__(self):
        self.observations = []

    def infer(self, observation, **kwargs):
        self.observations.append(observation)
        actions = np.arange(70, dtype=np.float32).reshape(10, 7) / 128 + len(self.observations)
        return {'actions': actions, 'policy_timing': {'infer_ms': 1.5}}


class NoInstrumentation:
    def __init__(self, policy):
        self.policy = policy

    def describe(self):
        return {'provider': 'test-fixture', 'checkpoint': 'untrained-test-fixture'}

    def validate(self, operation, targets):
        raise ValueError('Baseline fixture has no internal tools')


@contextmanager
def tool_server():
    from brainscore.experiments.backends.openpi_remote import OpenPIToolServer
    policy = ChunkPolicy()
    with listening(OpenPIToolServer(policy, NoInstrumentation(policy)).handler) as port:
        yield policy, 'ws://127.0.0.1:%d' % port


class Simulator:
    """Fake LIBERO environment: success after a fixed number of policy steps."""

    def __init__(self, done_at, fail_at=None, delay=0.0):
        self.done_at, self.fail_at, self.delay, self.t = done_at, fail_at, delay, 0

    def _observation(self):
        return {'agentview_image': np.full((8, 8, 3), self.t % 255, dtype=np.uint8),
                'robot0_eye_in_hand_image': np.zeros((8, 8, 3), dtype=np.uint8),
                'robot0_eef_pos': np.full(3, 0.1 * self.t), 'robot0_eef_quat': np.array([0., 0., 0., 1.]),
                'robot0_gripper_qpos': np.zeros(2)}

    def reset(self):
        self.t = 0

    def set_init_state(self, state):
        return self._observation()

    def step(self, action):
        time.sleep(self.delay)
        self.t += 1
        policy_step = self.t - 10
        if policy_step == self.fail_at:
            raise RuntimeError('simulator step failed')
        return self._observation(), 0.0, policy_step == self.done_at, {}

    def close(self):
        pass


def upstream_module(fail_at=None, delay=0.0):
    """The pinned eval_libero source, unmodified, around a fake benchmark and simulator."""
    path = OPENPI / 'examples/libero/main.py'
    tree = ast.parse(path.read_text())
    keep = [node for node in tree.body
            if (isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in ('Args', 'eval_libero', '_quat2axisangle'))
            or (isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None)
                in ('LIBERO_DUMMY_ACTION', 'LIBERO_ENV_RESOLUTION'))]
    from openpi_client import image_tools
    module = types.ModuleType('fake_openpi_libero')
    sys.modules[module.__name__] = module
    tasks = [SimpleNamespace(name=name, language='task ' + name) for name in TASKS]
    suite = SimpleNamespace(n_tasks=len(tasks), get_task=lambda i: tasks[i],
                            get_task_init_states=lambda i: [np.zeros(4)] * 50)
    module.__dict__.update(
        collections=collections, dataclasses=dataclasses, logging=logging, math=math,
        pathlib=pathlib, np=np, image_tools=image_tools, tqdm=SimpleNamespace(tqdm=lambda x: x),
        imageio=SimpleNamespace(mimwrite=lambda *args, **kwargs: None),
        benchmark=SimpleNamespace(get_benchmark_dict=lambda: {'libero_spatial': lambda: suite}),
        _get_libero_env=lambda task, resolution, seed: (
            Simulator(TASKS[task.name], fail_at if task.name == 'a' else None, delay), task.language))
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module, path


def evaluate_in_process(out, port, **upstream):
    module, source = upstream_module(**upstream)
    args = Namespace(out=out, route='umi', suite='libero_spatial', trials=1, port=port,
                     connect_timeout=60.0, infer_timeout=30.0, control_timeout=30.0,
                     finish_timeout=60.0)
    try:
        example('evaluate').run(module, source, args)
    except SystemExit as exit:
        return exit.code
    return 0


@contextmanager
def tool_bridge(module, tmp_path, idle_timeout=30.0):
    with tool_server() as (policy, upstream):
        bridge = module.Bridge(upstream, tmp_path / 'bridge', 'baseline', suite='libero_spatial',
                               trials_per_task=1, horizon=5, idle_timeout=idle_timeout)
        with listening(bridge.handler) as port:
            yield SimpleNamespace(policy=policy, bridge=bridge, port=port,
                                  uri='ws://127.0.0.1:%d' % port)


def wait_until(condition, timeout=30):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError('condition not reached')
        time.sleep(0.05)


def manifest(tmp_path, connection=1):
    return json.loads((tmp_path / 'bridge' / ('connection-%d' % connection) / 'experiment.json').read_text())


def record_events(experiment):
    from brainscore.run_record import RunRecord
    return list(RunRecord(experiment).events())


def acknowledgments(module, monkeypatch, tmp_path, drop=()):
    """Record each acknowledgment with the experiment status at the moment it is sent."""
    sent, send = [], module._best_effort

    def spy(socket, codec, message):
        sent.append((message['umi/finish_ack'], manifest(tmp_path)['status']))
        if message['umi/finish_ack'] not in drop:
            send(socket, codec, message)
    monkeypatch.setattr(module, '_best_effort', spy)
    return sent


def start_for(client):
    return {'umi/start': {'version': 'brainscore.libero.receipt.v1', 'run_id': client.run_id,
                          'suite': 'libero_spatial', 'trials_per_task': 1, 'execution_horizon': 5,
                          'num_steps_wait': 10, 'max_steps': 220, 'tasks': list(TASKS)}}


def libero_request():
    return {'observation/image': np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
            'observation/wrist_image': np.full((4, 4, 3), 7, dtype=np.uint8),
            'observation/state': np.arange(8, dtype=np.float32) / 3, 'prompt': 'task a'}


def test_max_steps_table_matches_pinned_evaluator():
    text = (OPENPI / 'examples/libero/main.py').read_text()
    pinned = {suite: int(steps) for suite, steps in
              re.findall(r'task_suite_name == "(\w+)":\s+max_steps = (\d+)', text)}
    assert pinned == example('evaluate').MAX_STEPS


def test_bounded_client_sends_upstream_bytes_and_parses_replies_identically():
    from openpi_client import msgpack_numpy
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from websockets.exceptions import ConnectionClosed
    received = []
    reply = {'actions': np.arange(21, dtype=np.float32).reshape(3, 7), 'policy_timing': {'infer_ms': 2.0}}

    def handler(websocket):
        websocket.send(msgpack_numpy.packb({'server': 'metadata'}))
        try:
            while True:
                raw = websocket.recv()
                received.append(raw)
                if msgpack_numpy.unpackb(raw)['prompt'] == 'fail':
                    websocket.send('policy failed')
                else:
                    websocket.send(msgpack_numpy.packb(reply))
        except ConnectionClosed:
            pass

    evaluate = example('evaluate')
    with listening(handler) as port:
        upstream = WebsocketClientPolicy('127.0.0.1', port)
        bounded = evaluate.BoundedClient('ws://127.0.0.1:%d' % port, connect_timeout=5, infer_timeout=5)
        try:
            assert bounded.metadata == upstream.get_server_metadata() and not bounded.receipt
            results = [upstream.infer(libero_request()), bounded.infer(libero_request(), {'index': 0})]
            assert received[0] == received[1]  # identical bytes, no annotation
            assert results[0].keys() == results[1].keys() == reply.keys()
            np.testing.assert_array_equal(results[0]['actions'], results[1]['actions'])
            assert results[1]['actions'].dtype == np.float32
            assert results[0]['policy_timing'] == results[1]['policy_timing']
            failing = dict(libero_request(), prompt='fail')
            with pytest.raises(RuntimeError, match='Error in inference server'):
                bounded.infer(failing)
            with pytest.raises(RuntimeError, match='not resent'):
                bounded.infer(libero_request())
            assert len(received) == 3
        finally:
            upstream._ws.close()
            bounded.close()


def test_lost_reply_times_out_once_and_is_never_resent():
    from openpi_client import msgpack_numpy
    from websockets.exceptions import ConnectionClosed
    received = []

    def handler(websocket):
        websocket.send(msgpack_numpy.packb({}))
        try:
            while True:
                received.append(websocket.recv())  # never replies
        except ConnectionClosed:
            pass

    evaluate = example('evaluate')
    with listening(handler) as port:
        client = evaluate.BoundedClient('ws://127.0.0.1:%d' % port, connect_timeout=5, infer_timeout=0.3)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            client.infer(libero_request())
        assert time.monotonic() - started < 5
        with pytest.raises(RuntimeError, match='not resent'):
            client.infer(libero_request())
        time.sleep(0.3)
        assert len(received) == 1
    port = free_port()
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        evaluate.BoundedClient('ws://127.0.0.1:%d' % port, connect_timeout=1.5, infer_timeout=1)
    assert time.monotonic() - started < 6


def test_tool_bridge_acknowledges_only_saved_complete_runs_and_compare_joins_records(tmp_path, monkeypatch):
    module, compare = example('serve_tool_bridge'), example('compare')
    sent = acknowledgments(module, monkeypatch, tmp_path)
    with tool_bridge(module, tmp_path) as running:
        assert evaluate_in_process(tmp_path / 'eval', running.port) == 0
        wait_until(lambda: not running.bridge.lock.locked())
    assert sent == [('complete', 'complete')]  # sent after Experiment.run saved the manifest
    evaluation, experiment = tmp_path / 'eval', tmp_path / 'bridge/connection-1'
    rows = [json.loads(line) for line in (evaluation / 'trials.jsonl').read_text().splitlines()]
    assert [(r['task'], r['ended_by'], r['calls'], r['last_executed'], r['full_calls']) for r in rows] == [
        ('a', 'success', 2, 2, 1), ('b', 'success', 2, 5, 2), ('c', 'step_limit', 44, 5, 44)]
    assert json.loads((evaluation / 'finish.json').read_text())['ack'] == 'complete'
    # Requests reach the policy without annotations; one inference per call.
    assert len(running.policy.observations) == sum(r['calls'] for r in rows) == 48
    assert all(set(o) == set(libero_request()) for o in running.policy.observations)
    result = compare.compare(evaluation, evaluation, reference_experiment=experiment,
                             umi_experiment=experiment)
    assert result['scope'] == {'reference': 'tool-bridge', 'umi': 'tool-bridge'}
    with pytest.raises(ValueError, match='pass its UMI experiment'):
        compare.compare(evaluation, evaluation)

    def tampered(change, name):
        copy = tmp_path / name
        shutil.copytree(evaluation, copy / 'eval')
        shutil.copytree(experiment, copy / 'experiment')
        change(copy / 'eval', copy / 'experiment')
        return copy / 'eval', copy / 'experiment'

    unacknowledged = tampered(lambda e, x: (e / 'finish.json').unlink(), 'unacknowledged')
    with pytest.raises(ValueError, match='not acknowledged'):
        compare.compare(evaluation, unacknowledged[0], reference_experiment=experiment,
                        umi_experiment=unacknowledged[1])
    assert compare.compare(evaluation, unacknowledged[0], reference_experiment=experiment,
                           umi_experiment=unacknowledged[1], reconcile_unacknowledged=True)['scope'][
        'umi'] == 'tool-bridge-reconciled-unacknowledged'

    def edit_report(e, x):
        report = json.loads((e / 'report.json').read_text())
        report['successes_note'] = 'edited'
        (e / 'report.json').write_text(json.dumps(report))

    def fail_manifest(e, x):
        value = json.loads((x / 'experiment.json').read_text())
        value['status'] = 'failed'
        (x / 'experiment.json').write_text(json.dumps(value))

    def edit_trials(e, x):
        rows = (e / 'trials.jsonl').read_text().splitlines()
        row = json.loads(rows[0])
        row['calls'] = 3
        (e / 'trials.jsonl').write_text('\n'.join([json.dumps(row)] + rows[1:]) + '\n')

    for change, message in [(edit_report, 'does not match this report'),
                            (fail_manifest, 'did not complete'), (edit_trials, 'differ')]:
        changed = tampered(change, change.__name__)
        with pytest.raises(ValueError, match=message):
            compare.compare(evaluation, changed[0], reference_experiment=experiment,
                            umi_experiment=changed[1], reconcile_unacknowledged=True)


def test_bridge_preserves_request_and_chunk_and_rejects_out_of_order_calls(tmp_path):
    module, evaluate = example('serve_tool_bridge'), example('evaluate')
    with tool_bridge(module, tmp_path) as running:
        client = evaluate.BoundedClient(running.uri, connect_timeout=10, infer_timeout=10)
        assert client.receipt and client.bridge['experiment_tools'] == 'baseline'
        assert client.control(start_for(client), 10) == {'umi/start_ack': 'accepted', 'run_id': client.run_id}
        result = client.infer(libero_request(), {'index': 0, 'task': 'a', 'trial': 0})
        np.testing.assert_array_equal(
            result['actions'], np.arange(70, dtype=np.float32).reshape(10, 7) / 128 + 1)
        assert result['actions'].dtype == np.float32 and result['policy_timing'] == {'infer_ms': 1.5}
        received = running.policy.observations[0]
        assert received['prompt'] == 'task a' and set(received) == set(libero_request())
        for key in ('observation/image', 'observation/wrist_image', 'observation/state'):
            np.testing.assert_array_equal(received[key], libero_request()[key])
            assert received[key].dtype == libero_request()[key].dtype
        with pytest.raises(RuntimeError, match='Bridge stopped the run'):
            client.infer(libero_request(), {'index': 1, 'task': 'c', 'trial': 0})  # skips task b
        wait_until(lambda: not running.bridge.lock.locked())
    assert manifest(tmp_path)['status'] == 'failed'
    assert len(running.policy.observations) == 1


def test_inconsistent_receipt_is_recorded_rejected_and_answered_with_failure(tmp_path):
    module, evaluate = example('serve_tool_bridge'), example('evaluate')
    with tool_bridge(module, tmp_path) as running:
        client = evaluate.BoundedClient(running.uri, connect_timeout=10, infer_timeout=10)
        client.control(start_for(client), 10)
        client.infer(libero_request(), {'index': 0, 'task': 'a', 'trial': 0})
        episode = {'task': 'a', 'trial': 0, 'calls': 2, 'full_calls': 0, 'last_executed': 2,
                   'executed_total': 7, 'wait_steps': 10, 'success': True, 'ended_by': 'success',
                   'prefix_match': True, 'execution_unknown': False, 'errors': 0}
        receipt = {'version': 'brainscore.libero.receipt.v1', 'run_id': client.run_id,
                   'report_sha256': '0' * 64, 'status': 'complete', 'episodes': [episode] * 3}
        reply = client.control({'umi/receipt': receipt}, 10)
        assert reply['umi/finish_ack'] == 'failed' and reply['run_id'] == client.run_id
        wait_until(lambda: not running.bridge.lock.locked())
    assert manifest(tmp_path)['status'] == 'failed'
    kinds = [event['kind'] for event in record_events(tmp_path / 'bridge/connection-1')]
    assert kinds.count('evaluator_receipt') == 1 and kinds.count('evaluator_start') == 1


def test_disconnect_or_idle_evaluator_fails_the_run_and_releases_the_bridge(tmp_path):
    module, evaluate = example('serve_tool_bridge'), example('evaluate')
    with tool_bridge(module, tmp_path, idle_timeout=0.5) as running:
        client = evaluate.BoundedClient(running.uri, connect_timeout=10, infer_timeout=10)
        client.control(start_for(client), 10)
        client.close()  # disconnect before any receipt
        wait_until(lambda: not running.bridge.lock.locked())
        idle = evaluate.BoundedClient(running.uri, connect_timeout=10, infer_timeout=10)
        wait_until(lambda: (tmp_path / 'bridge/connection-2/experiment.json').exists()
                   and manifest(tmp_path, 2)['status'] != 'running')
        wait_until(lambda: not running.bridge.lock.locked())
        idle.close()
    assert manifest(tmp_path, 1)['status'] == 'failed'
    assert manifest(tmp_path, 2)['status'] == 'failed'
    assert manifest(tmp_path, 2)['error']['type'] == 'TimeoutError'


def test_swallowed_simulator_error_fails_evaluator_and_bridge(tmp_path, monkeypatch):
    module = example('serve_tool_bridge')
    sent = acknowledgments(module, monkeypatch, tmp_path)
    with tool_bridge(module, tmp_path) as running:
        assert evaluate_in_process(tmp_path / 'eval', running.port, fail_at=3) == 1
        wait_until(lambda: not running.bridge.lock.locked())
    report = json.loads((tmp_path / 'eval/report.json').read_text())
    first = json.loads((tmp_path / 'eval/trials.jsonl').read_text().splitlines()[0])
    assert report['status'] == 'failed' and first['execution_unknown'] and first['ended_by'] == 'error'
    assert json.loads((tmp_path / 'eval/finish.json').read_text())['ack'] == 'failed'
    assert sent == [('failed', 'failed')]


def test_record_finalization_failure_sends_only_a_failed_acknowledgment(tmp_path, monkeypatch):
    module = example('serve_tool_bridge')

    class FailingRecorder(module.RecordInputsOutputs):
        @contextmanager
        def attach(self, context):
            with super().attach(context):
                yield
            raise OSError('record could not be finalized')
    monkeypatch.setattr(module, 'RecordInputsOutputs', FailingRecorder)
    sent = acknowledgments(module, monkeypatch, tmp_path)
    with tool_bridge(module, tmp_path) as running:
        assert evaluate_in_process(tmp_path / 'eval', running.port) == 1
        wait_until(lambda: not running.bridge.lock.locked())
    assert json.loads((tmp_path / 'eval/report.json').read_text())['status'] == 'complete'
    assert json.loads((tmp_path / 'eval/finish.json').read_text())['ack'] == 'failed'
    assert sent == [('failed', 'failed')]
    assert manifest(tmp_path)['error']['type'] == 'OSError'


def test_lost_completion_acknowledgment_needs_explicit_reconciliation(tmp_path, monkeypatch):
    module, compare = example('serve_tool_bridge'), example('compare')
    acknowledgments(module, monkeypatch, tmp_path, drop=('complete',))
    with tool_bridge(module, tmp_path) as running:
        assert evaluate_in_process(tmp_path / 'eval', running.port) == 1
        wait_until(lambda: not running.bridge.lock.locked())
    evaluation, experiment = tmp_path / 'eval', tmp_path / 'bridge/connection-1'
    assert manifest(tmp_path)['status'] == 'complete'
    assert json.loads((evaluation / 'finish.json').read_text())['ack'] in ('closed', 'timeout')
    events = record_events(experiment)
    calls = sum(e['kind'] == 'input' and e['payload']['method'] == 'process' for e in events)
    assert [e['kind'] for e in events].count('evaluator_receipt') == 1
    assert calls == len(running.policy.observations) == 48  # nothing resent after the receipt
    with pytest.raises(ValueError, match='not acknowledged'):
        compare.compare(evaluation, evaluation, reference_experiment=experiment, umi_experiment=experiment)
    assert compare.compare(evaluation, evaluation, reference_experiment=experiment,
                           umi_experiment=experiment, reconcile_unacknowledged=True)['scope'] == {
        'reference': 'tool-bridge-reconciled-unacknowledged',
        'umi': 'tool-bridge-reconciled-unacknowledged'}


class CloseFailsSocket:
    """An evaluator connection that never sends and whose close raises."""

    def __init__(self):
        self.sent, self.closed = [], 0

    def send(self, data):
        from openpi_client import msgpack_numpy
        self.sent.append(msgpack_numpy.unpackb(data))

    def recv(self, timeout=None):
        raise TimeoutError('evaluator idle')

    def close(self, *args, **kwargs):
        self.closed += 1
        raise OSError('close failed')


def test_failed_close_releases_the_bridge_and_keeps_the_run_error(tmp_path):
    module = example('serve_tool_bridge')
    with tool_server() as (policy, upstream):
        bridge = module.Bridge(upstream, tmp_path / 'bridge', 'baseline', suite='libero_spatial',
                               trials_per_task=1, horizon=5)
        socket = CloseFailsSocket()
        with pytest.raises(TimeoutError, match='evaluator idle'):  # not the close OSError
            bridge.handler(socket)
        assert not bridge.lock.locked() and socket.closed == 1
        assert socket.sent[-1]['umi/finish_ack'] == 'failed'
        assert manifest(tmp_path)['status'] == 'failed'
        assert manifest(tmp_path)['error']['type'] == 'TimeoutError'
        # The released bridge accepts the next evaluator.
        with pytest.raises(TimeoutError):
            bridge.handler(CloseFailsSocket())
        assert manifest(tmp_path, 2)['status'] == 'failed' and not bridge.lock.locked()


def bridge_process(tmp_path, upstream, port):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    command = [sys.executable, str(ROOT / 'examples/libero/serve_tool_bridge.py'), '--upstream', upstream,
               '--port', str(port), '--out', str(tmp_path / 'bridge'), '--mode', 'baseline',
               '--suite', 'libero_spatial', '--trials-per-task', '1', '--execution-horizon', '5']
    log = (tmp_path / 'bridge.log').open('w')
    return subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)


def evaluator_process(tmp_path, port, mode):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    log = (tmp_path / 'evaluator.log').open('w')
    return subprocess.Popen([sys.executable, __file__, str(tmp_path / 'eval'), str(port), mode],
                            env=env, stdout=log, stderr=subprocess.STDOUT)


@contextmanager
def stopped_at_exit(process):
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.mark.parametrize('mode', ['clean', 'killed'])
def test_separate_evaluator_and_bridge_processes(tmp_path, mode):
    compare = example('compare')
    port = free_port()
    with tool_server() as (policy, upstream), stopped_at_exit(bridge_process(tmp_path, upstream, port)):
        with stopped_at_exit(evaluator_process(tmp_path, port, mode)) as evaluator:
            if mode == 'clean':
                assert evaluator.wait(timeout=90) == 0, (tmp_path / 'evaluator.log').read_text()
            else:
                wait_until(lambda: len(policy.observations) >= 2, timeout=60)
                evaluator.kill()
                evaluator.wait()
        wait_until(lambda: (tmp_path / 'bridge/connection-1/experiment.json').exists()
                   and manifest(tmp_path)['status'] != 'running', timeout=30)
    evaluation, experiment = tmp_path / 'eval', tmp_path / 'bridge/connection-1'
    if mode == 'clean':
        assert manifest(tmp_path)['status'] == 'complete'
        assert compare.compare(evaluation, evaluation, reference_experiment=experiment,
                               umi_experiment=experiment)['scope']['umi'] == 'tool-bridge'
    else:
        assert manifest(tmp_path)['status'] == 'failed'
        with pytest.raises((OSError, ValueError)):
            compare.compare(evaluation, evaluation, reference_experiment=experiment,
                            umi_experiment=experiment)


if __name__ == '__main__':
    # Evaluator subprocess for test_separate_evaluator_and_bridge_processes.
    logging.basicConfig(level=logging.WARNING)
    out, bridge_port, run_mode = sys.argv[1:]
    sys.exit(evaluate_in_process(Path(out), int(bridge_port), delay=0.05 if run_mode == 'killed' else 0.0))
