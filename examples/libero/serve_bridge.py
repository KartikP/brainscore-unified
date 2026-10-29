"""Serve a recorded reference or UMI route using OpenPI's websocket protocol.

The record is a transport capture, not evidence that an evaluation finished.
It closes 'complete' only if exactly one evaluator session connected and closed
its connection before the bridge was stopped, and every request it sent has a
recorded reply. Anything else (no session, an evaluator still connected, or a
stop, crash or policy failure during a call) closes it 'failed'. An evaluator
that crashes still closes its connection, so qualification also requires that
run's completed report; see replay.verify_capture.

On SIGTERM or SIGINT the outcome is fixed at once (complete only for a closed
session), and the record is saved from the event loop before the server shuts
down. A stop during a blocked policy call interrupts that call, which fails the
record. STOP_GRACE_SECONDS, counted from the first signal, covers saving the
record and closing connections. If they have not finished by then, the process
exits with status 3 and does nothing else: a record not yet saved keeps its
'open' manifest, which RunRecord.inspect reports as incomplete.
"""
import argparse
import asyncio
from copy import deepcopy
import logging
import os
import signal
import threading

from brainscore_core.events import EnvironmentStep
from brainscore_core.model_interface import BrainScoreModel
from brainscore.harnesses.libero_policy import LiberoChunkPolicy
from brainscore.run_record import RunRecorder

CAPTURE = 'brainscore.libero.transport-capture.v1'
# Operational bound from the first stop signal to exit, including saving the record and
# closing connections; not a measured budget. A large log can need longer to hash.
STOP_GRACE_SECONDS = 5.0


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
        self.calling = False  # true while a request is being answered

    def infer(self, request):
        self.calling = True
        try:
            return self._infer(request)
        except BaseException:
            # Includes a stop during a call: the input may have no reply. Never retried.
            self.recorder.close(failed=True)
            raise
        finally:
            self.calling = False

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

    stop = {}  # the outcome, fixed at the first stop signal

    class OneSession(WebsocketPolicyServer):
        """Serve one evaluator connection per capture; refuse every later one.

        Overrides the per-connection handler and run() of the pinned upstream
        server (OPENPI_REVISION in evaluate.py). Health checks never reach it.
        """
        sessions = 0
        connected = False
        loop = task = None

        async def run(self):
            self.task = asyncio.current_task()
            self.loop = asyncio.get_running_loop()
            if stop:
                self.loop.call_soon(finish)
            await super().run()

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

    def finish():
        # Runs once, in the event loop: never inside a policy call or a record write.
        if stop.get('finishing'):
            return  # run() and on_stop() can both schedule it in one narrow window
        stop['finishing'] = True
        try:
            recorder.close(failed=stop['failed'])
        finally:
            server.task.cancel()

    def on_stop(*_):
        # Python runs this between any two bytecodes of the main thread. An exception
        # raised here can be discarded (for example inside a weakref callback), so
        # the stop is scheduled on the event loop rather than relying on it.
        if not stop:
            stop['failed'] = server.sessions != 1 or server.connected
            # The emergency exit does no I/O, so it cannot fail, block or race a write.
            stop['watchdog'] = threading.Timer(STOP_GRACE_SECONDS, os._exit, args=(3,))
            stop['watchdog'].daemon = True
            stop['watchdog'].start()
            if server.loop is not None and not server.loop.is_closed():
                server.loop.call_soon_threadsafe(finish)
        if server.loop is None or policy.calling:
            raise KeyboardInterrupt  # interrupt a blocked call, or stop before serving

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        try:
            recorder.close(failed=stop.get('failed', True))
        finally:
            try:
                # Close without a handshake: the policy may still be answering an
                # interrupted call, and its connection threads must not delay exit.
                upstream._ws.close_socket()
            finally:
                if 'watchdog' in stop:
                    stop['watchdog'].cancel()
                    stop['watchdog'].join()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
