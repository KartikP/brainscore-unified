"""Remote scope lifecycle tests; no trained policy or paid compute required."""
from contextlib import contextmanager
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from brainscore.experiments.backends.openpi_remote import (
    OpenPIToolServer, OpenPIPolicyClient, RemoteOpenPIInstrumentation, PROTOCOL,
)
from brainscore_core.events import Selection


class FixturePolicy:
    def __init__(self):
        self.scopes = []
        self.started = threading.Event()
        self.proceed = threading.Event()
        self.proceed.set()

    def infer(self, observation, **kwargs):
        self.started.set()
        self.proceed.wait(timeout=5)
        value = np.asarray(observation['value'], dtype=np.float32)
        for operation, targets, receive in self.scopes:
            if operation == 'ablate':
                value = np.zeros_like(value)
            else:
                receive('layer', {'array': value.copy(), 'denoising_step': 0, 'dtype': 'float32'})
        if observation.get('fail'):
            raise ValueError('inference failed')
        return {'actions': value}


class FixtureInstrumentation:
    def __init__(self, policy):
        self.policy = policy

    def describe(self):
        return {'provider': 'test-fixture', 'checkpoint': 'untrained-test-fixture'}

    def validate(self, operation, targets):
        if operation not in ('record', 'ablate') or any(t.layer != 'layer' for t in targets):
            raise ValueError('Unsupported selection')

    @contextmanager
    def record(self, targets, receive):
        self.policy.scopes.append(('record', targets, receive))
        try:
            yield
        finally:
            self.policy.scopes.pop()

    @contextmanager
    def ablate(self, targets):
        self.policy.scopes.append(('ablate', targets, None))
        try:
            yield
        finally:
            self.policy.scopes.pop()


def request(operation='infer', plan=None, **kwargs):
    return dict(protocol=PROTOCOL, request_id='test-request', operation=operation,
                plan=plan or [], observation={'value': [1, 2]}, **kwargs)


def scope(operation):
    return {'operation': operation, 'targets': [{'layer': 'layer', 'indices': None}]}


@pytest.fixture
def server():
    policy = FixturePolicy()
    return OpenPIToolServer(policy, FixtureInstrumentation(policy))


def test_configuration_only_validates_and_failed_inference_cleans_up(server):
    server.handle(request('configure', [scope('ablate')]))
    assert server.policy.scopes == []
    failed = request(plan=[scope('ablate')])
    failed['observation']['fail'] = True
    with pytest.raises(ValueError, match='inference failed'):
        server.handle(failed)
    assert server.policy.scopes == []
    np.testing.assert_array_equal(server.handle(request())['output']['actions'], [1, 2])


@pytest.mark.parametrize('plan', [None, {}, [{'operation': 'ablate'}], [scope('unknown')]])
def test_malformed_plan_rejected_before_inference(server, plan):
    value = request()
    value['plan'] = plan
    with pytest.raises(ValueError):
        server.handle(value)
    assert not server.policy.started.is_set()


def load_codec():
    # Source checkout is optional for the websocket transport tests.
    import os
    from pathlib import Path
    import sys
    source = os.environ.get('OPENPI_SOURCE')
    if source:
        sys.path.insert(0, str(Path(source) / 'packages/openpi-client/src'))
    return pytest.importorskip('openpi_client.msgpack_numpy')


@contextmanager
def listening(server):
    load_codec()
    from websockets.sync.server import serve
    with serve(server.handler, '127.0.0.1', 0, compression=None,
               max_size=server.max_message_bytes) as websocket_server:
        thread = threading.Thread(target=websocket_server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'ws://127.0.0.1:{websocket_server.socket.getsockname()[1]}'
        finally:
            websocket_server.shutdown()
            thread.join(timeout=5)


def test_loopback_record_ablate_ack_and_cleanup(server):
    with listening(server) as uri, OpenPIPolicyClient(uri) as client:
        instrumentation = RemoteOpenPIInstrumentation(client)
        instrumentation.validate('record', ['layer'])
        before, after = [], []
        with instrumentation.record(['layer'], lambda _, v: before.append(v)):
            with instrumentation.ablate([Selection(layer='layer')]):
                with instrumentation.record(['layer'], lambda _, v: after.append(v)):
                    np.testing.assert_array_equal(client.infer({'value': [1, 2]})['actions'], [0, 0])
        np.testing.assert_array_equal(before[0]['array'], [1, 2])
        np.testing.assert_array_equal(after[0]['array'], [0, 0])
        assert before[0]['request_id'] == after[0]['request_id']
        assert before[0]['inference_index'] == 1
        assert server.policy.scopes == []
        np.testing.assert_array_equal(client.infer({'value': [1, 2]})['actions'], [1, 2])


def test_disconnect_during_inference_cleans_before_next_request(server):
    codec = load_codec()
    from websockets.sync.client import connect
    server.policy.proceed.clear()
    with listening(server) as uri:
        socket = connect(uri, compression=None)
        socket.recv()
        socket.send(codec.packb(request(plan=[scope('ablate')])))
        assert server.policy.started.wait(timeout=5)
        socket.close()
        server.policy.proceed.set()
        with OpenPIPolicyClient(uri) as next_client:
            np.testing.assert_array_equal(next_client.infer({'value': [3, 4]})['actions'], [3, 4])
        assert server.policy.scopes == []


def test_record_budget_failure_removes_hooks(server):
    load_codec()
    server.max_message_bytes = 32
    with pytest.raises(ValueError, match='budget'):
        server.handle(request(plan=[scope('ablate'), scope('record')]))
    assert server.policy.scopes == []


def test_mismatched_response_closes_client():
    codec = load_codec()
    class WrongServer:
        max_message_bytes = 1024
        def handler(self, socket):
            socket.send(codec.packb({'protocol': PROTOCOL, 'instrumentation': {}}))
            socket.recv()
            socket.send(codec.packb({'protocol': PROTOCOL, 'request_id': 'wrong'}))
    with listening(WrongServer()) as uri, OpenPIPolicyClient(uri) as client:
        with pytest.raises(RuntimeError, match='Mismatched'):
            client.infer({'value': [1]})
        assert client._closed


def test_missing_configuration_ack_closes_client():
    codec = load_codec()
    class NoAck:
        max_message_bytes = 1024
        def handler(self, socket):
            socket.send(codec.packb({'protocol': PROTOCOL, 'instrumentation': {}}))
            incoming = codec.unpackb(socket.recv())
            socket.send(codec.packb(dict(incoming, accepted=False)))
    with listening(NoAck()) as uri, OpenPIPolicyClient(uri) as client:
        with pytest.raises(RuntimeError, match='acknowledge'):
            RemoteOpenPIInstrumentation(client).validate('record', ['layer'])
        assert client._closed and client._scopes == []


def test_client_callback_failure_cleans_scopes(server):
    with listening(server) as uri, OpenPIPolicyClient(uri) as client:
        def fail(*args):
            raise ValueError('recorder failed')
        instrumentation = RemoteOpenPIInstrumentation(client)
        with pytest.raises(ValueError, match='recorder failed'):
            with instrumentation.ablate(['layer']), instrumentation.record(['layer'], fail):
                client.infer({'value': [1]})
        assert client._closed and client._scopes == [] and server.policy.scopes == []


def test_server_serializes_clients(server):
    from concurrent.futures import ThreadPoolExecutor
    server.policy.proceed.clear()
    with listening(server) as uri:
        def ablated():
            with OpenPIPolicyClient(uri) as client:
                with RemoteOpenPIInstrumentation(client).ablate(['layer']):
                    return client.infer({'value': [1]})['actions']
        def baseline():
            with OpenPIPolicyClient(uri) as client:
                return client.infer({'value': [2]})['actions']
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(ablated)
            assert server.policy.started.wait(timeout=5)
            second = pool.submit(baseline)
            server.policy.proceed.set()
            np.testing.assert_array_equal(first.result(timeout=5), [0])
            np.testing.assert_array_equal(second.result(timeout=5), [2])
        assert server.policy.scopes == []
