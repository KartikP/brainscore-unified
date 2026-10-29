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
import signal
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
            yield process
        except BaseException as error:
            failure = error
            raise
        else:
            failure = None
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                # A forced kill is never a clean stop: say so, with the server's own log.
                message = ('%s did not exit within 10 s of SIGTERM and was killed; log tail:\n%s'
                           % (Path(command[1]).name, log.read_text()[-4000:]))
                if failure is None:
                    raise AssertionError(message)
                failure.add_note(message)


POLICY_FIXTURE = '''
import sys, time
import numpy as np
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
class Policy:
    def infer(self, request):
        if sys.argv[2] == 'block':
            time.sleep(60)
        value = float(request['observation/state'][0])
        return {'actions': np.arange(70, dtype=np.float32).reshape(10, 7) / 128 + value}
WebsocketPolicyServer(Policy(), host='127.0.0.1', port=int(sys.argv[1])).serve_forever()
'''


def transport_env(monkeypatch):
    for path in (OPENPI / 'src', OPENPI / 'packages/openpi-client/src'):
        monkeypatch.syspath_prepend(str(path))
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(OPENPI / 'src'), str(OPENPI / 'packages/openpi-client/src'),
                                       *sys.path])
    return env


@contextmanager
def policy_server(env, tmp_path, mode='answer'):
    port = free_port()
    with server([sys.executable, '-c', POLICY_FIXTURE, str(port), mode], port, env,
                tmp_path / ('policy-%s.log' % mode)):
        yield port


# Runs serve_bridge.py so that GET /stop-in-finalizer delivers SIGTERM to the bridge while
# Python is inside a weakref callback, where an exception raised by the signal handler is
# discarded ("Exception ignored"). This reproduces a CI failure on Linux at random timing.
STOP_IN_FINALIZER = '''
import os, runpy, signal, sys, weakref
import openpi.serving.websocket_policy_server as upstream
health = upstream._health_check
class Finalized:
    pass
def check(connection, request):
    if request.path == '/stop-in-finalizer':
        def callback(_):
            os.kill(os.getpid(), signal.SIGTERM)
            sum(range(100000))  # the Python signal handler runs during these bytecodes
            print('stop signal handled inside a weakref callback', file=sys.stderr, flush=True)
        item = Finalized()
        reference = weakref.ref(item, callback)
        del item
    return health(connection, request)
upstream._health_check = check
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''


@contextmanager
def capture_bridge(env, tmp_path, name, route, upstream_port, prefix=(), process=None):
    port, log = free_port(), tmp_path / (name + '.log')
    command = [sys.executable, *prefix, str(ROOT / 'examples/libero/serve_bridge.py'), '--route', route,
               '--upstream-port', str(upstream_port), '--port', str(port), '--out', str(tmp_path / name)]
    with server(command, port, env, log) as started:
        if process is not None:
            process.append(started)
        yield port, log
    # server() has stopped the bridge with SIGTERM, as docker stop does.


def sessions_closed(log, count=1):
    return lambda: len(re.findall(r'Connection from .* closed', log.read_text())) >= count


def capture_manifest(path):
    from brainscore.run_record import RunRecord
    return RunRecord(path).manifest


def test_reference_and_umi_captures_need_the_evaluators_report(tmp_path, monkeypatch):
    env = transport_env(monkeypatch)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    replay, toolbox = example('replay'), example('toolbox_calls')
    with policy_server(env, tmp_path) as policy_port:
        for route in ('reference', 'umi'):
            with capture_bridge(env, tmp_path, route + '-record', route, policy_port) as (port, log):
                # The pinned evaluator loop with a fake simulator: 2 + 2 + 44 calls.
                assert evaluate_in_process(tmp_path / (route + '-eval'), port, route=route) == 0
                wait_until(sessions_closed(log))
            assert capture_manifest(tmp_path / (route + '-record'))['status'] == 'complete'
        record, report = tmp_path / 'reference-record', tmp_path / 'reference-eval'
        with capture_bridge(env, tmp_path, 'replay-record', 'umi', policy_port) as (port, log):
            client = WebsocketClientPolicy('127.0.0.1', port)
            try:
                with pytest.raises(ValueError, match='completed transport'):
                    replay.compare_record(record, client, report=tmp_path / 'umi-eval')
                result = replay.compare_record(record, client, report=report,
                                               suite='libero_spatial', trials_per_task=1)
                assert result['passed'] and result['calls'] == 48 and result['max_absolute_error'] == 0
            finally:
                client._ws.close()
            wait_until(sessions_closed(log))
        assert capture_manifest(tmp_path / 'replay-record')['event_count'] == 96
        # The same external policy is also usable through the common toolbox runner.
        client = WebsocketClientPolicy('127.0.0.1', policy_port)
        try:
            outcome = toolbox.run_calls(client, record, tmp_path / 'toolbox', report=report)
            assert outcome.value['calls'] == 48 and outcome.value['exact']
            assert outcome.manifest['status'] == 'complete'
        finally:
            client._ws.close()


def test_stop_during_a_call_fails_the_capture(tmp_path, monkeypatch):
    env = transport_env(monkeypatch)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    with policy_server(env, tmp_path, 'block') as policy_port:
        with capture_bridge(env, tmp_path, 'record', 'reference', policy_port) as (port, log):
            client = WebsocketClientPolicy('127.0.0.1', port)
            threading.Thread(target=lambda: client.infer(libero_request()), daemon=True).start()
            events = tmp_path / 'record/events.jsonl'
            wait_until(lambda: events.read_bytes().count(b'\n') == 1)  # input saved; reply pending
    manifest = capture_manifest(tmp_path / 'record')
    assert manifest['status'] == 'failed' and manifest['event_count'] == 1


@pytest.mark.parametrize('state', ['no_session', 'evaluator_connected'])
def test_stop_without_a_closed_session_fails_the_capture(tmp_path, monkeypatch, state):
    env = transport_env(monkeypatch)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    with policy_server(env, tmp_path) as policy_port:
        with capture_bridge(env, tmp_path, 'record', 'reference', policy_port) as (port, log):
            if state == 'evaluator_connected':
                client = WebsocketClientPolicy('127.0.0.1', port)
                client.infer(libero_request())  # one full exchange, then the stop
    manifest = capture_manifest(tmp_path / 'record')
    assert manifest['status'] == 'failed'
    assert manifest['event_count'] == (2 if state == 'evaluator_connected' else 0)


def test_capture_serves_one_evaluator_session(tmp_path, monkeypatch):
    env = transport_env(monkeypatch)
    import websockets
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    with policy_server(env, tmp_path) as policy_port:
        with capture_bridge(env, tmp_path, 'record', 'reference', policy_port) as (port, log):
            first = WebsocketClientPolicy('127.0.0.1', port)
            first.infer(libero_request())
            with pytest.raises(websockets.ConnectionClosed) as refused:
                WebsocketClientPolicy('127.0.0.1', port)  # while the first is connected
            assert refused.value.rcvd.code == 1013
            first.infer(libero_request())  # the first session is unaffected
            first._ws.close()
            wait_until(sessions_closed(log))
            with pytest.raises(websockets.ConnectionClosed):
                WebsocketClientPolicy('127.0.0.1', port)  # and after it ended
    manifest = capture_manifest(tmp_path / 'record')
    assert manifest['status'] == 'complete' and manifest['event_count'] == 4


def test_stop_signal_lost_in_a_finalizer_still_saves_the_capture(tmp_path, monkeypatch):
    env = transport_env(monkeypatch)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    record = tmp_path / 'record'
    with policy_server(env, tmp_path) as policy_port:
        with capture_bridge(env, tmp_path, 'record', 'reference', policy_port,
                            prefix=('-c', STOP_IN_FINALIZER)) as (port, log):
            client = WebsocketClientPolicy('127.0.0.1', port)
            client.infer(libero_request())
            client._ws.close()
            wait_until(sessions_closed(log))
            try:
                urllib.request.urlopen('http://127.0.0.1:%d/stop-in-finalizer' % port, timeout=5).close()
            except OSError:
                pass  # not a websocket request; only the signal matters
            wait_until(lambda: capture_manifest(record)['status'] != 'open', timeout=8)
            assert 'stop signal handled inside a weakref callback' in log.read_text()
    manifest = capture_manifest(record)
    assert manifest['status'] == 'complete' and manifest['event_count'] == 2


def test_server_helper_reports_a_forced_kill(tmp_path):
    ignores_sigterm = (
        'import http.server, signal, sys\n'
        'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
        'class Health(http.server.BaseHTTPRequestHandler):\n'
        '    def do_GET(self):\n'
        '        self.send_response(200); self.end_headers()\n'
        'http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Health).serve_forever()\n')
    port = free_port()
    with pytest.raises(AssertionError, match='did not exit within 10 s of SIGTERM and was killed'):
        with server([sys.executable, '-c', ignores_sigterm, str(port)], port, dict(os.environ),
                    tmp_path / 'stubborn.log'):
            pass


# Runs serve_bridge.py with injected shutdown faults, selected by BRIDGE_FAULT:
#   close-hangs   saving the record never returns
#   close-fails   saving the record raises OSError
#   slow-upstream closing the upstream socket takes 1 s (a window after the loop closed)
# Each step it reaches is printed, so a test can see what ran.
BRIDGE_FAULTS = '''
import os, runpy, sys, time
from brainscore.run_record import RunRecorder
import websockets.sync.connection as connection
fault, close, close_socket = os.environ['BRIDGE_FAULT'], RunRecorder.close, connection.Connection.close_socket
def record_close(self, **kwargs):
    if fault == 'close-hangs':
        time.sleep(60)
    if fault == 'close-fails':
        raise OSError('injected record failure')
    return close(self, **kwargs)
def upstream_close(self):
    print('upstream socket closed', file=sys.stderr, flush=True)
    if fault == 'slow-upstream':
        time.sleep(1)
    return close_socket(self)
RunRecorder.close, connection.Connection.close_socket = record_close, upstream_close
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''


@contextmanager
def faulty_bridge_session(tmp_path, monkeypatch, fault):
    """One complete evaluator session on a bridge with an injected fault; yields the process."""
    env = dict(transport_env(monkeypatch), BRIDGE_FAULT=fault)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    process = []
    with policy_server(env, tmp_path) as policy_port:
        with capture_bridge(env, tmp_path, 'record', 'reference', policy_port,
                            prefix=('-c', BRIDGE_FAULTS), process=process) as (port, log):
            client = WebsocketClientPolicy('127.0.0.1', port)
            client.infer(libero_request())
            client._ws.close()
            wait_until(sessions_closed(log))
            yield process[0], log


def test_emergency_exit_is_unconditional_when_saving_the_record_hangs(tmp_path, monkeypatch):
    from brainscore.run_record import RunRecord
    with faulty_bridge_session(tmp_path, monkeypatch, 'close-hangs') as (process, log):
        started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=9) == 3  # the stop budget, then exit; nothing else ran
        assert 4 < time.monotonic() - started < 9
    assert capture_manifest(tmp_path / 'record')['status'] == 'open'
    assert RunRecord.inspect(tmp_path / 'record')['status'] == 'incomplete'


def test_record_failure_still_closes_upstream_without_emergency_exit(tmp_path, monkeypatch):
    with faulty_bridge_session(tmp_path, monkeypatch, 'close-fails') as (process, log):
        started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        code = process.wait(timeout=9)
        elapsed = time.monotonic() - started
    text = log.read_text()
    assert 'upstream socket closed' in text and 'OSError: injected record failure' in text
    assert code == 1 and elapsed < 3  # the error propagates; no emergency exit
    assert capture_manifest(tmp_path / 'record')['status'] == 'open'


def test_repeated_stop_after_the_loop_closed_is_ignored(tmp_path, monkeypatch):
    with faulty_bridge_session(tmp_path, monkeypatch, 'slow-upstream') as (process, log):
        process.send_signal(signal.SIGTERM)
        wait_until(lambda: 'upstream socket closed' in log.read_text(), timeout=5)
        process.send_signal(signal.SIGTERM)  # while main() is cleaning up; the loop is closed
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=9) == 0
    text = log.read_text()
    assert 'Traceback' not in text and 'Event loop is closed' not in text, text
    manifest = capture_manifest(tmp_path / 'record')
    assert manifest['status'] == 'complete' and manifest['event_count'] == 2


def test_remote_qualification_reports_unchanged_actions_without_gating_them(tmp_path, monkeypatch):
    from tests.test_libero_policy import UnitInstrumentation, UnitPolicy, request
    import brainscore.experiments as experiments
    monkeypatch.setattr(experiments, 'OpenPIInstrumentation', UnitInstrumentation)
    module = example('qualify_remote_tools')
    report = module.qualify(UnitPolicy(), [request(), request()], tmp_path / 'q', 'unit-fixture')
    assert report['status'] == 'passed' and report['effects'] == {'ablation_changed_actions': False}
    assert report['selected_unit_zeroed'] and report['selected_unit_was_nonzero']
    with pytest.raises(RuntimeError, match='not silenced'):
        module.qualify(UnitPolicy(ablation_works=False), [request()], tmp_path / 'broken', 'unit-fixture')
    assert json.loads((tmp_path / 'broken/qualification.json').read_text())['status'] == 'failed'


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
    """Serve handler; on exit, also wait for every connection handler to finish.

    websockets' shutdown() neither tracks nor joins connection threads, so a
    handler may still be closing its socket (under a deadline timer) after the
    client is gone. Joining them makes cleanup checks deterministic.
    """
    from websockets.sync.server import serve
    connections = []

    def tracked(websocket):
        connections.append(threading.current_thread())
        handler(websocket)

    with serve(tracked, '127.0.0.1', 0, compression=None, max_size=64 * 1024 * 1024,
               logger=QUIET) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.socket.getsockname()[1]
        finally:
            server.shutdown()
            thread.join(timeout=10)
            deadline = time.monotonic() + 15
            for connection in connections:
                connection.join(timeout=max(deadline - time.monotonic(), 0))
            alive = [connection.name for connection in connections if connection.is_alive()]
            assert not alive, 'Connection handlers still running after 15 s: %s' % alive


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


def evaluate_in_process(out, port, route='umi', **upstream):
    module, source = upstream_module(**upstream)
    args = Namespace(out=out, route=route, suite='libero_spatial', trials=1, port=port,
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

    def spy(socket, codec, message, timeout):
        sent.append((message['umi/finish_ack'], manifest(tmp_path)['status']))
        if message['umi/finish_ack'] not in drop:
            send(socket, codec, message, timeout)
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


# A separately prepared interpreter (e.g. Python 3.8 with numpy, pytest, websockets,
# msgpack and Pillow) can run the evaluator side; the default is this interpreter.
EVALUATOR_PYTHON = os.environ.get('UMI_EVALUATOR_PYTHON') or sys.executable


def evaluator_process(tmp_path, port, mode):
    # The evaluator never imports brainscore; another interpreter gets only the pinned client.
    path = (sys.path if EVALUATOR_PYTHON == sys.executable
            else [str(OPENPI / 'packages/openpi-client/src')])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(path))
    log = (tmp_path / 'evaluator.log').open('w')
    return subprocess.Popen([EVALUATOR_PYTHON, __file__, str(tmp_path / 'eval'), str(port), mode],
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
        version = subprocess.check_output([EVALUATOR_PYTHON, '-c', 'import sys; print(sys.version)'],
                                          text=True).strip()
        assert json.loads((evaluation / 'report.json').read_text())['python'] == version
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


# Bounded writes and closes on the evaluator connection.

def test_evaluator_request_to_a_stalled_bridge_is_bounded_and_never_resent():
    from openpi_client import msgpack_numpy
    from tests.test_openpi_remote import assert_one_blocked_write, counted_sends, stalled_peer, timers
    evaluate = example('evaluate')
    with stalled_peer(msgpack_numpy.packb({})) as uri:
        client = evaluate.BoundedClient(uri, connect_timeout=5, infer_timeout=1.0, close_timeout=1.0)
        sends = counted_sends(client._ws)
        with pytest.raises(TimeoutError):
            client.infer(dict(libero_request(), **{'observation/image': np.zeros(4 << 20, np.uint8)}))
        assert_one_blocked_write(sends, 1.0)
        assert client.broken and not client._ws.recv_events_thread.is_alive()
        with pytest.raises(RuntimeError, match='not resent'):
            client.infer(libero_request())
        assert len(sends) == 1 and timers() == []
    for value in (0, -1.0, float('inf'), float('nan'), True):
        with pytest.raises(ValueError):
            evaluate.BoundedClient(uri, connect_timeout=5, infer_timeout=5, close_timeout=value)


class StalledEvaluator:
    """An evaluator that stopped reading: a send blocks until the socket is shut down.

    With stall_close, close blocks the same way; otherwise it returns at once.
    """

    def __init__(self, stall_close=False):
        self.shut = threading.Event()
        self.close_socket = self.shut.set  # the deadline's termination step
        self.sent, self.closes, self.stall_close = [], 0, stall_close

    def _block(self):
        if not self.shut.wait(timeout=30):
            raise AssertionError('deadline never fired')
        raise OSError('socket shut down')

    def send(self, data):
        from openpi_client import msgpack_numpy
        self.sent.append(msgpack_numpy.unpackb(data))
        self._block()

    def recv(self, timeout=None):
        raise TimeoutError('evaluator idle')

    def close(self, **kwargs):
        self.closes += 1
        if self.stall_close:
            self._block()


def test_stalled_send_fails_the_run_and_releases_the_bridge(tmp_path):
    from tests.test_openpi_remote import timers
    module = example('serve_tool_bridge')
    with tool_server() as (policy, upstream):
        bridge = module.Bridge(upstream, tmp_path / 'bridge', 'baseline', suite='libero_spatial',
                               trials_per_task=1, horizon=5, idle_timeout=0.3)
        evaluator, started = StalledEvaluator(), time.monotonic()
        with pytest.raises(TimeoutError, match='deadline'):
            bridge.handler(evaluator)
        assert time.monotonic() - started < 10 and not bridge.lock.locked()
        # The hello blocked once; the failed acknowledgment was attempted once, never resent.
        assert [list(message)[0] for message in evaluator.sent] == ['experiment_tools', 'umi/finish_ack']
        assert evaluator.sent[1]['umi/finish_ack'] == 'failed' and evaluator.closes == 1
        assert manifest(tmp_path)['status'] == 'failed' and policy.observations == []
    # After the fixture has joined its connection handlers, including their bounded closes.
    assert timers() == []


def test_unresponsive_close_releases_the_bridge_and_keeps_the_run_error(tmp_path, monkeypatch):
    from brainscore.experiments.backends import socket_deadline
    from tests.test_openpi_remote import timers
    monkeypatch.setattr(socket_deadline, 'CLOSE_TIMEOUT', 0.3)
    module = example('serve_tool_bridge')
    with tool_server() as (policy, upstream):
        bridge = module.Bridge(upstream, tmp_path / 'bridge', 'baseline', suite='libero_spatial',
                               trials_per_task=1, horizon=5)
        evaluator = StalledEvaluator(stall_close=True)
        evaluator.send = lambda data: evaluator.sent.append(data)  # sends go through
        started = time.monotonic()
        with pytest.raises(TimeoutError, match='evaluator idle'):  # not the close timeout
            bridge.handler(evaluator)
        assert time.monotonic() - started < 10 and not bridge.lock.locked()
        assert evaluator.closes == 1 and evaluator.shut.is_set()
        assert manifest(tmp_path)['status'] == 'failed'
        # A second evaluator arriving while the bridge is busy is turned away within the bound.
        bridge.lock.acquire()
        try:
            busy = StalledEvaluator(stall_close=True)
            bridge.handler(busy)
            assert busy.closes == 1 and busy.shut.is_set() and busy.sent == []
        finally:
            bridge.lock.release()
    # After the fixture has joined its connection handlers, including their bounded closes.
    assert timers() == []
