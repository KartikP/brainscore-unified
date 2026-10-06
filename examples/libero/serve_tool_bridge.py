"""Connect the official LIBERO evaluator to the OpenPI tool server.

Run one evaluator per bridge. The evaluator owns observations, scheduling,
resets and scoring; Brain-Score records calls and applies the selected tools.
"""

import argparse
from pathlib import Path
import threading
from brainscore.experiments import (
    Experiment,
    CallableProtocol,
    RecordInputsOutputs,
    RecordActivity,
    Ablate,
    OpenPIPolicyClient,
    RemoteOpenPIInstrumentation,
)
from brainscore.harnesses.libero_policy import LiberoChunkPolicy
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.events import EnvironmentStep, Selection


class Bridge:
    def __init__(self, upstream, output, mode):
        self.upstream, self.output, self.mode = upstream, Path(output), mode
        self.lock = threading.Lock()
        self.connection = 0

    def handler(self, socket):
        from openpi_client import msgpack_numpy as codec
        from websockets.exceptions import ConnectionClosed

        if not self.lock.acquire(blocking=False):
            socket.close(code=1013, reason='Dedicated evaluator already connected')
            return
        try:
            self.connection += 1
            socket.send(codec.packb({'experiment_tools': self.mode}))
            with OpenPIPolicyClient(self.upstream) as client:
                subject = BrainScoreModel(
                    'pi05-libero', action_fn=LiberoChunkPolicy(client)
                )

                def evaluate(model, context):
                    calls = 0
                    while True:
                        try:
                            raw = socket.recv()
                        except ConnectionClosed:
                            return {'calls': calls}
                        request = codec.unpackb(raw)
                        response = model.process(
                            EnvironmentStep(
                                observation=request,
                                instruction=request['prompt'],
                                step_num=calls,
                            )
                        )
                        result = dict(
                            response.metadata['policy_output'], actions=response.action
                        )
                        socket.send(codec.packb(result))
                        calls += 1

                target = Selection(layer='action_out_proj', indices=[0])
                tools = [RecordInputsOutputs()]
                if self.mode == 'ablation':
                    tools.extend(
                        [
                            RecordActivity([target], when='before', name='before'),
                            Ablate([target]),
                        ]
                    )
                if self.mode != 'baseline':
                    tools.append(RecordActivity([target], name='after'))
                Experiment(
                    subject=subject,
                    protocol=CallableProtocol('libero-evaluator', evaluate),
                    tools=tools,
                    instrumentation=RemoteOpenPIInstrumentation(client),
                    output_dir=self.output / f'connection-{self.connection}',
                    metadata={
                        'mode': self.mode,
                        'evaluator_owns_resets_and_scoring': True,
                    },
                ).run()
        finally:
            self.lock.release()
            socket.close()


if __name__ == '__main__':
    from websockets.sync.server import serve

    p = argparse.ArgumentParser()
    p.add_argument('--upstream', default='ws://127.0.0.1:8000')
    p.add_argument('--port', type=int, default=8001)
    p.add_argument('--out', required=True)
    p.add_argument(
        '--mode', choices=['baseline', 'recording', 'ablation'], required=True
    )
    a = p.parse_args()
    bridge = Bridge(a.upstream, a.out, a.mode)
    with serve(
        bridge.handler, '127.0.0.1', a.port, compression=None, max_size=64 * 1024 * 1024
    ) as server:
        server.serve_forever()
