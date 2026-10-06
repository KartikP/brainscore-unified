"""Request-scoped OpenPI tools over a trusted, private websocket connection.

Configuration acknowledgments validate a plan; they never attach model hooks.
Hooks exist only inside one locked inference request and are removed before its
reply. A disconnected client cannot leave an ablation active for later calls.
"""
from contextlib import contextmanager, ExitStack
import threading
import uuid

from brainscore_core.events import Selection

PROTOCOL = 'brainscore.openpi.tools.v1'
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _codec():
    from openpi_client import msgpack_numpy
    return msgpack_numpy


def _targets(values):
    if not isinstance(values, list) or not 1 <= len(values) <= 64:
        raise ValueError('Each scope needs 1–64 targets')
    targets = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {'layer', 'indices'}:
            raise ValueError('Targets require layer and indices')
        targets.append(Selection(layer=value['layer'], indices=value['indices']))
    return targets


class OpenPIToolServer:
    """Serialize inference on one policy; bind to loopback or a private tunnel.

    No authentication/TLS or physical-robot safety is provided here. Closing a
    connection does not cancel a JAX computation already running. Its scopes are
    still removed before another request can use the policy.
    """
    def __init__(self, policy, instrumentation, *, idle_timeout=300,
                 max_message_bytes=MAX_MESSAGE_BYTES):
        if instrumentation.policy is not policy:
            raise ValueError('Server and instrumentation must use the same policy')
        self.policy, self.instrumentation = policy, instrumentation
        self.idle_timeout, self.max_message_bytes = idle_timeout, max_message_bytes
        self._lock = threading.Lock()
        self._inference_index = 0

    def _validate_plan(self, plan):
        if not isinstance(plan, list) or len(plan) > 32:
            raise ValueError('Plan must be a list of at most 32 tool scopes')
        parsed = []
        for scope in plan:
            if not isinstance(scope, dict) or set(scope) != {'operation', 'targets'}:
                raise ValueError('A scope requires operation and targets')
            targets = _targets(scope['targets'])
            self.instrumentation.validate(scope['operation'], targets)
            parsed.append((scope['operation'], targets))
        return parsed

    def handle(self, request):
        """Process one decoded request (also useful for transport-independent tests)."""
        if not isinstance(request, dict) or request.get('protocol') != PROTOCOL:
            raise ValueError('Unsupported tool protocol')
        request_id = request.get('request_id')
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 64:
            raise ValueError('A short request_id is required')
        operation = request.get('operation')
        if operation not in ('configure', 'infer'):
            raise ValueError('Unsupported request operation')
        plan = self._validate_plan(request.get('plan'))
        reply = {'protocol': PROTOCOL, 'request_id': request_id, 'operation': operation}
        if operation == 'configure':
            return dict(reply, accepted=True)
        if not self._lock.acquire(timeout=self.idle_timeout):
            raise TimeoutError('Policy is busy')
        try:
            self._inference_index += 1
            inference_index = self._inference_index
            activity, size = [], 0

            def receiver(scope_index):
                def receive(target, value):
                    nonlocal size
                    record = {'scope': scope_index, 'target': target, 'value': dict(
                        value, request_id=request_id, inference_index=inference_index)}
                    size += len(_codec().packb(record))
                    if size > self.max_message_bytes // 2:
                        raise ValueError('Activity exceeds the request budget; select fewer units or steps')
                    activity.append(record)
                return receive

            with ExitStack() as stack:
                for index, (op, targets) in enumerate(plan):
                    scope = (self.instrumentation.record(targets, receiver(index)) if op == 'record'
                             else self.instrumentation.ablate(targets))
                    stack.enter_context(scope)
                kwargs = {'noise': request['noise']} if 'noise' in request else {}
                output = self.policy.infer(request['observation'], **kwargs)
            # All hooks are detached before serialization or network operations.
            return dict(reply, output=output, activity=activity, inference_index=inference_index)
        finally:
            self._lock.release()

    def handler(self, websocket):
        from websockets.exceptions import ConnectionClosed
        codec = _codec()
        websocket.send(codec.packb({'protocol': PROTOCOL,
                                    'instrumentation': self.instrumentation.describe()}))
        try:
            while True:
                raw = websocket.recv(timeout=self.idle_timeout)
                request = None
                try:
                    if not isinstance(raw, bytes):
                        raise ValueError('Binary requests are required')
                    request = codec.unpackb(raw)
                    reply = self.handle(request)
                    packed = codec.packb(reply)
                    if len(packed) > self.max_message_bytes:
                        raise ValueError('Response exceeds the message budget')
                    websocket.send(packed)
                except ConnectionClosed:
                    raise
                except Exception as error:
                    # Do not send tracebacks or server paths to the client.
                    websocket.send(codec.packb({
                        'protocol': PROTOCOL,
                        'request_id': request.get('request_id') if isinstance(request, dict) else None,
                        'error': f'{type(error).__name__}: {error}',
                    }))
        except (ConnectionClosed, TimeoutError):
            pass
        finally:
            websocket.close()

    def serve(self, host='127.0.0.1', port=8001):
        from websockets.sync.server import serve
        with serve(self.handler, host, port, compression=None,
                   max_size=self.max_message_bytes) as server:
            server.serve_forever()


class OpenPIPolicyClient:
    """OpenPI-style infer client with separate activity delivery to attached tools.

    Use one connection per experiment thread. Requests are never retried: retrying
    would advance a stateful policy's RNG and could repeat a robot action.
    """
    def __init__(self, uri, *, timeout=300, max_message_bytes=MAX_MESSAGE_BYTES):
        from websockets.sync.client import connect
        self._socket = connect(uri, compression=None, max_size=max_message_bytes,
                               open_timeout=timeout)
        self.timeout = timeout
        self._closed = False
        self._owner = threading.get_ident()
        self._scopes = []
        self._provider = None
        self._busy = False
        try:
            self.metadata = _codec().unpackb(self._socket.recv(timeout=timeout))
            if self.metadata.get('protocol') != PROTOCOL:
                raise ValueError('Server does not support OpenPI experiment tools')
        except BaseException:
            self.close()
            raise

    def _check_thread(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('Use the client on its connection thread')

    def _plan(self):
        return [scope[0] for scope in self._scopes]

    def _request(self, operation, **payload):
        self._check_thread()
        if self._closed or self._busy:
            raise RuntimeError('Client is closed or already handling a request')
        request_id = uuid.uuid4().hex
        request = dict(payload, protocol=PROTOCOL, request_id=request_id, operation=operation)
        self._busy = True
        try:
            self._socket.send(_codec().packb(request))
            reply = _codec().unpackb(self._socket.recv(timeout=self.timeout))
            if reply.get('protocol') != PROTOCOL or reply.get('request_id') != request_id:
                raise RuntimeError('Mismatched response; connection closed')
            if 'error' in reply:
                raise RuntimeError(reply['error'])
            if reply.get('operation') != operation:
                raise RuntimeError('Mismatched operation; connection closed')
            if operation == 'configure' and reply.get('accepted') is not True:
                raise RuntimeError('Server did not acknowledge the tool configuration')
            return reply
        except BaseException:
            self.close()
            raise
        finally:
            self._busy = False

    def infer(self, observation, *, noise=None):
        kwargs = {} if noise is None else {'noise': noise}
        response = self._request('infer', plan=self._plan(), observation=observation, **kwargs)
        try:
            if (not isinstance(response.get('output'), dict)
                    or not isinstance(response.get('activity'), list)):
                raise RuntimeError('Malformed inference response')
            for record in response['activity']:
                index = record['scope']
                if (not isinstance(index, int) or not 0 <= index < len(self._scopes)
                        or self._scopes[index][0]['operation'] != 'record'):
                    raise RuntimeError('Unexpected activity scope')
                targets = self._scopes[index][0]['targets']
                if (record['target'] not in [target['layer'] for target in targets]
                        or record['value']['request_id'] != response['request_id']
                        or record['value']['inference_index'] != response['inference_index']):
                    raise RuntimeError('Activity does not match the inference request')
            for record in response['activity']:
                self._scopes[record['scope']][1](record['target'], record['value'])
        except BaseException:
            self.close()
            raise
        return response['output']

    def close(self):
        if not self._closed:
            self._closed = True
            self._socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class RemoteOpenPIInstrumentation:
    """Use RecordActivity/Ablate with an OpenPIPolicyClient on another process."""
    def __init__(self, client):
        if client._provider is not None:
            raise ValueError('Client already has an instrumentation provider')
        self.client = client
        client._provider = self

    def describe(self):
        return dict(self.client.metadata['instrumentation'], provider=type(self).__name__,
                    transport=PROTOCOL, scope='one inference request')

    def validate(self, operation, targets):
        self.client._request('configure', plan=[self._scope(operation, targets)])

    @staticmethod
    def _scope(operation, targets):
        selections = [Selection(layer=t) if isinstance(t, str) else t for t in targets]
        return {'operation': operation, 'targets': [
            {'layer': s.layer, 'indices': None if s.indices is None else list(s.indices)}
            for s in selections]}

    def record(self, targets, receive):
        if not callable(receive):
            raise TypeError('receive must be callable')
        return self._attach('record', targets, receive)

    def ablate(self, targets):
        return self._attach('ablate', targets, None)

    @contextmanager
    def _attach(self, operation, targets, receive):
        self.client._check_thread()
        scope = self._scope(operation, targets)
        self.client._request('configure', plan=self.client._plan() + [scope])
        self.client._scopes.append((scope, receive))
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self.client._scopes.pop()
            # The server never retains the plan, even when detach cannot be sent.
            if not self.client._closed:
                try:
                    self.client._request('configure', plan=self.client._plan())
                except Exception:
                    if not failed:
                        raise
