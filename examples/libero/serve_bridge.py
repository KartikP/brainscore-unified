"""Serve a recorded reference or UMI route using OpenPI's websocket protocol.

The record is a transport capture, not evidence that an evaluation finished.
It closes 'complete' only if exactly one evaluator session connected and closed
its connection before the bridge was stopped, and every request it sent has a
recorded reply. Anything else (no session, an evaluator still connected, or a
stop, crash or policy failure during a call) closes it 'failed'. An evaluator
that crashes still closes its connection, so qualification also requires that
run's completed report; see replay.verify_capture.
"""
import argparse
from copy import deepcopy
import logging
import signal

from brainscore_core.events import EnvironmentStep
from brainscore_core.model_interface import BrainScoreModel
from brainscore.harnesses.libero_policy import LiberoChunkPolicy
from brainscore.run_record import RunRecorder

CAPTURE = 'brainscore.libero.transport-capture.v1'


def capture_metadata(route, upstream):
    return {'route': route, 'upstream': upstream, 'capture': CAPTURE,
            'completion': 'transport capture only; evaluator completion requires its report',
            'evaluation': 'unqualified-libero-candidate'}


class RecordedPolicy:
    """One serial stream of policy queries; upstream owns inference RNG state.

    No reset RPC exists in the upstream websocket protocol. Restart the policy
    server before each matched run. No recurrent-policy support is implied.
    """

    def __init__(self, policy, *, route, recorder):
        if route not in ('reference', 'umi'):
            raise ValueError('Expected reference or umi route')
        self.policy = policy
        self.route = route
        self.recorder = recorder
        self.subject = BrainScoreModel('pi05-libero', policy=LiberoChunkPolicy(policy))
        self.calls = 0

    def infer(self, request):
        try:
            return self._infer(request)
        except BaseException:
            # Includes a stop during a call: the input may have no reply. Never retried.
            self.recorder.close(failed=True)
            raise

    def _infer(self, request):
        self.recorder.record('input', deepcopy(request))
        if self.route == 'reference':
            result = self.policy.infer(deepcopy(request))
        else:
            step = EnvironmentStep(
                observation=request, instruction=request['prompt'], step_num=self.calls,
                context={'protocol': 'openpi-libero-preprocessed-chunk'},
            )
            response = self.subject.process(step)
            result = dict(response.metadata['policy_output'], actions=response.action)
        self.recorder.record('output', deepcopy(result))
        self.calls += 1
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=('reference', 'umi'), required=True)
    parser.add_argument('--upstream-port', type=int, default=8000)
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    class OneSession(WebsocketPolicyServer):
        """Serve one evaluator connection per capture; refuse every later one.

        Overrides the per-connection handler of the pinned upstream server
        (OPENPI_REVISION in evaluate.py). Health checks never reach it.
        """
        sessions = 0
        connected = False

        async def _handler(self, websocket):
            if self.sessions:
                await websocket.close(code=1013, reason='This capture already served its evaluator')
                return
            self.sessions, self.connected = 1, True
            await super()._handler(websocket)
            # Reached only when the client closed the connection: not on a policy
            # error, and not when a stop cancels this handler.
            self.connected = False

    upstream = WebsocketClientPolicy('127.0.0.1', args.upstream_port)
    metadata = capture_metadata(args.route, upstream.get_server_metadata())
    recorder = RunRecorder(args.out, metadata=metadata)
    policy = RecordedPolicy(upstream, route=args.route, recorder=recorder)
    server = OneSession(policy, host='127.0.0.1', port=args.port, metadata=metadata)

    def stop(*_):
        raise KeyboardInterrupt

    # Also replaces asyncio's SIGINT handling, which would close the connection
    # first and make an interrupted session look like one the client ended.
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    failed = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        failed = server.sessions != 1 or server.connected
    finally:
        recorder.close(failed=failed)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
