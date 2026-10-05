"""Opt-in local websocket test using pinned OpenPI code and an untrained fixture.

Set OPENPI_SOURCE to a checkout; no checkpoint, simulator, or GPU is used.
"""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not os.environ.get('OPENPI_SOURCE'), reason='Requires pinned OpenPI source')]


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
