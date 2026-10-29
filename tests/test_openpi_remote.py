"""Remote scope lifecycle tests; no trained policy or paid compute required."""
from contextlib import contextmanager
import threading
import time
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


@pytest.mark.parametrize('field,value,message', [
    ('protocol', 'unknown', 'protocol'),
    ('request_id', '', 'request_id'),
    ('request_id', 'x' * 65, 'request_id'),
    ('request_id', None, 'request_id'),
    ('operation', 'reset-all', 'operation'),
    ('plan', [scope('record')] * 33, 'at most 32'),
    ('plan', [{'operation': 'record', 'targets': []}], '1–64 targets'),
    ('plan', [{'operation': 'record', 'targets': [{'layer': 'layer'}]}], 'layer and indices'),
])
def test_invalid_requests_never_execute_policy(server, field, value, message):
    invalid = request()
    invalid[field] = value
    with pytest.raises(ValueError, match=message):
        server.handle(invalid)
    assert not server.policy.started.is_set()
    assert server.policy.scopes == []
    np.testing.assert_array_equal(server.handle(request())['output']['actions'], [1, 2])


def test_busy_policy_rejects_request_without_attaching_tools(server):
    server.idle_timeout = 0
    server._lock.acquire()
    try:
        with pytest.raises(TimeoutError, match='busy'):
            server.handle(request(plan=[scope('ablate')]))
        assert not server.policy.started.is_set()
        assert server.policy.scopes == []
    finally:
        server._lock.release()
    np.testing.assert_array_equal(server.handle(request())['output']['actions'], [1, 2])


def test_server_rejects_instrumentation_for_another_policy():
    with pytest.raises(ValueError, match='same policy'):
        OpenPIToolServer(FixturePolicy(), FixtureInstrumentation(FixturePolicy()))


class ReplyServer:
    """A wire-level peer that acknowledges tools but can return a damaged reply."""
    max_message_bytes = 1024 * 1024

    def __init__(self, change_reply):
        self.change_reply = change_reply
        self.calls = 0

    def handler(self, socket):
        from websockets.exceptions import ConnectionClosed
        codec = load_codec()
        socket.send(codec.packb({'protocol': PROTOCOL, 'instrumentation': {}}))
        try:
            while True:
                incoming = codec.unpackb(socket.recv())
                reply = {
                    'protocol': PROTOCOL,
                    'request_id': incoming['request_id'],
                    'operation': incoming['operation'],
                    'accepted': True,
                }
                if incoming['operation'] == 'infer':
                    self.calls += 1
                    activity = {
                        'scope': 0,
                        'target': 'layer',
                        'value': {'request_id': incoming['request_id'], 'inference_index': 1},
                    }
                    reply.update(output={'actions': [1]}, activity=[activity], inference_index=1)
                    self.change_reply(reply)
                socket.send(codec.packb(reply))
        except ConnectionClosed:
            pass


@pytest.mark.parametrize('change,message', [
    (lambda reply: reply.update(operation='configure'), 'Mismatched operation'),
    (lambda reply: reply.update(error='policy failed'), 'policy failed'),
    (lambda reply: reply.update(output=None), 'Malformed inference'),
    (lambda reply: reply.update(activity={}), 'Malformed inference'),
    (lambda reply: reply['activity'][0].update(scope=-1), 'Unexpected activity scope'),
    (lambda reply: reply['activity'][0].update(scope=1), 'Unexpected activity scope'),
    (lambda reply: reply['activity'][0].update(scope='0'), 'Unexpected activity scope'),
    (lambda reply: reply['activity'][0].update(target='other'), 'does not match'),
    (lambda reply: reply['activity'][0]['value'].update(request_id='other'), 'does not match'),
    (lambda reply: reply['activity'][0]['value'].update(inference_index=2), 'does not match'),
])
def test_invalid_reply_closes_connection_without_delivering_activity(change, message):
    peer = ReplyServer(change)
    seen = []
    with listening(peer) as uri, OpenPIPolicyClient(uri) as client:
        instrumentation = RemoteOpenPIInstrumentation(client)
        with pytest.raises(RuntimeError, match=message):
            with instrumentation.record(['layer'], lambda target, value: seen.append(value)):
                client.infer({'value': [1]})
        assert seen == [] and client._closed and client._scopes == []
        with pytest.raises(RuntimeError, match='closed'):
            client.infer({'value': [1]})
    assert peer.calls == 1  # Failed requests must never be retried automatically.


def test_all_activity_is_validated_before_any_callback():
    def damage(reply):
        reply['activity'].append(dict(reply['activity'][0], target='wrong-layer'))
    seen = []
    with listening(ReplyServer(damage)) as uri, OpenPIPolicyClient(uri) as client:
        with pytest.raises(RuntimeError, match='does not match'):
            with RemoteOpenPIInstrumentation(client).record(['layer'], lambda *args: seen.append(args)):
                client.infer({'value': [1]})
        assert seen == []


def test_activity_cannot_be_delivered_to_an_ablation_scope():
    with listening(ReplyServer(lambda reply: None)) as uri, OpenPIPolicyClient(uri) as client:
        with pytest.raises(RuntimeError, match='Unexpected activity scope'):
            with RemoteOpenPIInstrumentation(client).ablate(['layer']):
                client.infer({'value': [1]})
        assert client._closed and client._scopes == []


def test_server_reports_policy_failure_and_next_client_is_unmodified(server):
    with listening(server) as uri:
        with OpenPIPolicyClient(uri) as client:
            with pytest.raises(RuntimeError, match='inference failed'):
                with RemoteOpenPIInstrumentation(client).ablate(['layer']):
                    client.infer({'value': [1], 'fail': True})
            assert client._closed and server.policy.scopes == []
        with OpenPIPolicyClient(uri) as fresh:
            np.testing.assert_array_equal(fresh.infer({'value': [2]})['actions'], [2])


def test_text_request_is_rejected_without_poisoning_connection(server):
    codec = load_codec()
    from websockets.sync.client import connect
    with listening(server) as uri, connect(uri, compression=None) as socket:
        socket.recv()
        socket.send('not a binary request')
        assert 'Binary requests' in codec.unpackb(socket.recv())['error']
        assert not server.policy.started.is_set()
        socket.send(codec.packb(request()))
        np.testing.assert_array_equal(codec.unpackb(socket.recv())['output']['actions'], [1, 2])


def test_client_rejects_foreign_thread_without_sending_request(server):
    from concurrent.futures import ThreadPoolExecutor
    with listening(server) as uri, OpenPIPolicyClient(uri) as client:
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(client.infer, {'value': [1]})
            with pytest.raises(RuntimeError, match='connection thread'):
                future.result(timeout=5)
        assert not server.policy.started.is_set()
        np.testing.assert_array_equal(client.infer({'value': [2]})['actions'], [2])


def test_invalid_provider_setup_leaves_client_usable(server):
    with listening(server) as uri, OpenPIPolicyClient(uri) as client:
        instrumentation = RemoteOpenPIInstrumentation(client)
        with pytest.raises(ValueError, match='already has'):
            RemoteOpenPIInstrumentation(client)
        with pytest.raises(TypeError, match='callable'):
            instrumentation.record(['layer'], None)
        assert client._scopes == [] and not server.policy.started.is_set()
        np.testing.assert_array_equal(client.infer({'value': [2]})['actions'], [2])


def test_unsupported_server_handshake_closes_connection():
    codec = load_codec()
    disconnected = threading.Event()

    class UnsupportedServer:
        max_message_bytes = 1024

        def handler(self, socket):
            from websockets.exceptions import ConnectionClosed
            socket.send(codec.packb({'protocol': 'other-service'}))
            try:
                socket.recv()
            except ConnectionClosed:
                disconnected.set()

    with listening(UnsupportedServer()) as uri:
        with pytest.raises(ValueError, match='does not support'):
            OpenPIPolicyClient(uri)
        assert disconnected.wait(timeout=5)


def test_oversized_response_cleans_tools_before_reporting_error(server, monkeypatch):
    original = server.policy.infer

    def large_output(observation, **kwargs):
        result = original(observation, **kwargs)
        return {'actions': np.tile(result['actions'], 4096)}

    server.max_message_bytes = 1024
    monkeypatch.setattr(server.policy, 'infer', large_output)
    with listening(server) as uri:
        with OpenPIPolicyClient(uri) as client:
            with pytest.raises(RuntimeError, match='Response exceeds'):
                with RemoteOpenPIInstrumentation(client).ablate(['layer']):
                    client.infer({'value': [1]})
            assert server.policy.scopes == [] and client._closed
        monkeypatch.setattr(server.policy, 'infer', original)
        with OpenPIPolicyClient(uri) as fresh:
            np.testing.assert_array_equal(fresh.infer({'value': [3]})['actions'], [3])


@pytest.mark.parametrize('body_fails', [False, True])
def test_detach_failure_preserves_original_error_and_clears_scope(body_fails):
    codec = load_codec()

    class RejectDetach:
        max_message_bytes = 1024

        def handler(self, socket):
            from websockets.exceptions import ConnectionClosed
            socket.send(codec.packb({'protocol': PROTOCOL, 'instrumentation': {}}))
            try:
                while True:
                    incoming = codec.unpackb(socket.recv())
                    reply = dict(incoming, accepted=True)
                    if not incoming['plan']:
                        reply['error'] = 'detach failed'
                    socket.send(codec.packb(reply))
            except ConnectionClosed:
                pass

    with listening(RejectDetach()) as uri, OpenPIPolicyClient(uri) as client:
        error, message = (ValueError, 'experiment failed') if body_fails else (RuntimeError, 'detach failed')
        with pytest.raises(error, match=message):
            with RemoteOpenPIInstrumentation(client).ablate(['layer']):
                if body_fails:
                    raise ValueError('experiment failed')
        assert client._scopes == [] and client._closed


# Transport deadlines (socket_deadline): sends, whole exchanges and closes.

class FakeConnection:
    """Stands in for a websockets connection whose peer stopped reading.

    send/close block until the deadline shuts the socket down, like a blocked
    sendall does, then fail the way the library does.
    """
    def __init__(self):
        self.shut = threading.Event()
        self.close_socket = self.shut.set  # the deadline's termination step
        self.sends = self.closes = 0

    def send(self, data):
        self.sends += 1
        self._block()

    def close(self, **kwargs):
        self.closes += 1
        self._block()

    def _block(self):
        if not self.shut.wait(timeout=10):
            raise AssertionError('deadline never fired')
        raise OSError('socket shut down')


def timers():
    return [thread for thread in threading.enumerate() if isinstance(thread, threading.Timer)]


@pytest.mark.parametrize('seconds', [0, -1, float('inf'), float('nan'), True, '1', None])
def test_invalid_timeouts_are_rejected_before_any_io(seconds):
    # Runs without websockets installed: validation precedes optional imports.
    from brainscore.experiments.backends import socket_deadline
    connection = FakeConnection()
    with pytest.raises(ValueError):
        with socket_deadline.deadline(connection, seconds):
            connection.send(b'never')
    assert connection.sends == 0 and timers() == []
    for kwargs in ({'timeout': seconds}, {'close_timeout': seconds}):
        with pytest.raises(ValueError):
            OpenPIPolicyClient('ws://127.0.0.1:1', **kwargs)
    policy = FixturePolicy()
    with pytest.raises(ValueError):
        OpenPIToolServer(policy, FixtureInstrumentation(policy), idle_timeout=seconds)


def test_deadline_aborts_a_blocked_send_once_and_chains_its_error():
    from brainscore.experiments.backends import socket_deadline
    connection, started = FakeConnection(), time.monotonic()
    with pytest.raises(TimeoutError) as raised:
        with socket_deadline.deadline(connection, 0.05):
            connection.send(b'request')
    assert isinstance(raised.value.__cause__, OSError)
    assert connection.sends == 1 and time.monotonic() - started < 5 and timers() == []


def test_finished_deadline_never_fires_later_and_keeps_unrelated_errors():
    from brainscore.experiments.backends import socket_deadline
    connection = FakeConnection()
    with socket_deadline.deadline(connection, 0.05) as remaining:
        assert 0 < remaining() <= 0.05
    with pytest.raises(KeyError):
        with socket_deadline.deadline(connection, 0.05):
            raise KeyError('unrelated')
    time.sleep(0.2)
    assert not connection.shut.is_set() and timers() == []


def test_exchange_finishing_as_the_deadline_fires_counts_as_timed_out():
    # The socket may already be shut down, so the connection cannot be reused.
    from brainscore.experiments.backends import socket_deadline
    connection = FakeConnection()
    with pytest.raises(TimeoutError):
        with socket_deadline.deadline(connection, 0.05):
            assert connection.shut.wait(timeout=5)  # reply 'arrives' just after the abort
    assert timers() == []


def test_close_is_bounded_and_never_raises():
    from brainscore.experiments.backends import socket_deadline
    stalled, started = FakeConnection(), time.monotonic()
    assert socket_deadline.close(stalled, 0.05) is False
    assert stalled.shut.is_set() and time.monotonic() - started < 5
    shut = []

    def fail(**kwargs):
        raise OSError('close failed')
    failing = SimpleNamespace(close=fail, close_socket=lambda: shut.append(True))
    assert socket_deadline.close(failing, 5) is False and shut == [True] and timers() == []


@contextmanager
def stalled_peer(hello, buffer=4096):
    """A real loopback websocket server that sends hello and then never reads.

    Sans-I/O over a raw socket: a websockets connection object would keep
    reading in its receiver thread. Small buffers make a few-MB message block
    the sender quickly.
    """
    import socket
    from websockets.server import ServerProtocol
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer)
    listener.bind(('127.0.0.1', 0))
    listener.listen()
    release, accepted = threading.Event(), []

    def serve():
        connection = listener.accept()[0]
        accepted.append(connection)
        protocol = ServerProtocol(max_size=None)
        while not (events := protocol.events_received()):
            protocol.receive_data(connection.recv(65536))
        protocol.send_response(protocol.accept(events[0]))
        protocol.send_binary(hello)
        connection.sendall(b''.join(protocol.data_to_send()))
        release.wait(timeout=30)  # never reads again

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f'ws://127.0.0.1:{listener.getsockname()[1]}'
    finally:
        release.set()
        thread.join(timeout=5)
        for connection in accepted + [listener]:
            connection.close()


@contextmanager
def stalled_client(port, message, buffer=4096):
    """A raw loopback websocket client that sends one message and never reads."""
    import socket
    from websockets.client import ClientProtocol
    from websockets.uri import parse_uri
    connection = socket.socket()
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer)
    connection.connect(('127.0.0.1', port))
    try:
        protocol = ClientProtocol(parse_uri(f'ws://127.0.0.1:{port}'), max_size=None)
        protocol.send_request(protocol.connect())
        connection.sendall(b''.join(protocol.data_to_send()))
        while not protocol.events_received():
            protocol.receive_data(connection.recv(1024))
        protocol.send_binary(message)
        connection.sendall(b''.join(protocol.data_to_send()))
        yield
    finally:
        connection.close()


def counted_sends(connection, buffer=4096):
    """Shrink the send buffer and record (size, seconds, returned) for each send.

    returned is False when the send raised: it never completed its write.
    """
    import socket
    connection.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer)
    sends, send = [], connection.send

    def counted(data):
        started, returned = time.monotonic(), False
        try:
            send(data)
            returned = True
        finally:
            sends.append((len(data), time.monotonic() - started, returned))
    connection.send = counted
    return sends


def assert_one_blocked_write(sends, timeout):
    # Exactly one send; it was still writing when the deadline cut it off.
    assert len(sends) == 1
    size, seconds, returned = sends[0]
    assert size > 4 << 20 and not returned and timeout * 0.9 <= seconds < timeout + 4


def test_client_request_to_a_stalled_server_is_bounded_and_never_resent():
    codec = load_codec()
    with stalled_peer(codec.packb({'protocol': PROTOCOL, 'instrumentation': {}})) as uri:
        client = OpenPIPolicyClient(uri, timeout=1.0, close_timeout=1.0)
        sends = counted_sends(client._socket)
        with pytest.raises(TimeoutError):
            client.infer({'value': np.zeros(4 << 20, dtype=np.uint8)})
        assert_one_blocked_write(sends, 1.0)
        assert client._closed and not client._socket.recv_events_thread.is_alive()
        with pytest.raises(RuntimeError, match='closed'):
            client.infer({'value': [1]})
        assert len(sends) == 1 and timers() == []


def test_close_meets_its_deadline_while_the_receive_queue_is_full():
    # A full queue pauses the receiver; websockets' own close_timeout is much longer.
    load_codec()
    from brainscore.experiments.backends import socket_deadline
    from websockets.sync.client import connect
    from websockets.sync.server import serve
    flooded, stop = threading.Event(), threading.Event()

    def handler(websocket):
        for _ in range(20):
            websocket.send(b'x')
        flooded.set()
        stop.wait(timeout=10)

    with serve(handler, '127.0.0.1', 0, close_timeout=0.2) as websocket_server:
        thread = threading.Thread(target=websocket_server.serve_forever, daemon=True)
        thread.start()
        connection = connect(f'ws://127.0.0.1:{websocket_server.socket.getsockname()[1]}',
                             max_queue=1, close_timeout=5)
        try:
            assert flooded.wait(timeout=5)
            time.sleep(0.1)
            started = time.monotonic()
            assert socket_deadline.close(connection, 0.1) is False
            assert time.monotonic() - started < 0.8
            connection.recv_events_thread.join(timeout=1)
            assert not connection.recv_events_thread.is_alive() and timers() == []
        finally:
            stop.set()
            websocket_server.shutdown()
            thread.join(timeout=5)


def test_completed_request_deadline_does_not_affect_the_next_request(server):
    with listening(server) as uri, OpenPIPolicyClient(uri, timeout=0.5) as client:
        np.testing.assert_array_equal(client.infer({'value': [1, 2]})['actions'], [1, 2])
        time.sleep(0.8)  # past the first request's deadline
        np.testing.assert_array_equal(client.infer({'value': [3, 4]})['actions'], [3, 4])
        assert timers() == []


def test_server_reply_to_a_client_that_stops_reading_is_bounded_and_frees_the_policy(server):
    import socket
    codec = load_codec()
    from websockets.sync.server import serve
    server.idle_timeout = 0.5
    finished, sends = threading.Event(), []

    def handler(websocket):
        sends.append(counted_sends(websocket))
        try:
            server.handler(websocket)
        finally:
            finished.set()

    with serve(handler, '127.0.0.1', 0, compression=None,
               max_size=server.max_message_bytes) as websocket_server:
        thread = threading.Thread(target=websocket_server.serve_forever, daemon=True)
        thread.start()
        port = websocket_server.socket.getsockname()[1]
        # The 4 MB float32 reply cannot fit in the buffers; the client never reads it.
        message = codec.packb(dict(request(), observation={'value': np.zeros(1 << 20, np.uint8)}))
        try:
            with stalled_client(port, message):
                started = time.monotonic()
                assert finished.wait(timeout=10) and time.monotonic() - started < 5
                hello, *replies = sends[0]
                assert hello[2]  # the small hello went out; the reply never completed
                assert_one_blocked_write(replies, 0.5)
                assert not server._lock.locked() and timers() == []
                with OpenPIPolicyClient(f'ws://127.0.0.1:{port}') as client:
                    np.testing.assert_array_equal(client.infer({'value': [5]})['actions'], [5])
        finally:
            websocket_server.shutdown()
            thread.join(timeout=5)
