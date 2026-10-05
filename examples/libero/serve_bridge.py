"""Serve a recorded reference or UMI route using OpenPI's websocket protocol."""
import argparse
from copy import deepcopy
import logging
import signal

from brainscore_core.events import EnvironmentStep
from brainscore_core.model_interface import BrainScoreModel
from brainscore.harnesses.libero_policy import LiberoChunkPolicy
from brainscore.run_record import RunRecorder


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
        self.subject = BrainScoreModel('pi05-libero', action_fn=LiberoChunkPolicy(policy))
        self.calls = 0

    def infer(self, request):
        try:
            return self._infer(request)
        except Exception:
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
    upstream = WebsocketClientPolicy('127.0.0.1', args.upstream_port)
    metadata = {'route': args.route, 'upstream': upstream.get_server_metadata(),
                'evaluation': 'unqualified-libero-candidate'}
    with RunRecorder(args.out, metadata=metadata) as recorder:
        policy = RecordedPolicy(upstream, route=args.route, recorder=recorder)
        # Bind to localhost. Run one evaluator or replay client at a time.
        def stop(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        try:
            WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port,
                                  metadata=metadata).serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
