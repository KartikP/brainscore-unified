"""Rerun saved OpenPI requests through UMI's common experiment runner.

Requires a freshly initialized policy server and the same checkpoint/runtime
profile as the source. This does not run LIBERO physics or reproduce task scores.
Never attach this client to a server currently running a benchmark: calls change
its random state. Internal activity/ablation needs a provider inside that server.
"""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
from brainscore.experiments import Experiment, CallableProtocol, RecordInputsOutputs
from brainscore.run_record import RunRecord
from brainscore.harnesses.libero_policy import LiberoChunkPolicy
from brainscore_core.events import EnvironmentStep
from brainscore_core.model_interface import BrainScoreModel


def _replay():
    path = Path(__file__).resolve().with_name('replay.py')
    spec = importlib.util.spec_from_file_location('libero_toolbox_replay', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_calls(policy, source, output_dir, *, report):
    """report is the evaluator output directory of the run that produced source."""
    replay = _replay()
    saved = RunRecord(source)
    evidence = replay.verify_capture(saved, report)
    calls = list(replay.pairs(saved, evidence['calls']))
    subject = BrainScoreModel('pi05-libero', policy=LiberoChunkPolicy(policy))
    def evaluate(model, context):
        differences = []
        for i, (request, output) in enumerate(calls):
            actual = model.process(EnvironmentStep(observation=request,
                instruction=request['prompt'], step_num=i)).action
            expected = np.asarray(output['actions'])
            if actual.shape != expected.shape or not np.isfinite(expected).all():
                raise ValueError('Invalid or mismatched action shape')
            differences.append(float(np.max(np.abs(actual.astype(float)-expected.astype(float)))))
        return {'calls': len(differences), 'max_absolute_error': max(differences),
                'exact': all(value == 0 for value in differences),
                'scope': 'saved inputs; no simulator or task success assessment'}
    result = Experiment(subject=subject,
        protocol=CallableProtocol('openpi-saved-requests', evaluate,
            metadata={'reset_policy': 'caller must initialize a fresh server'}),
        tools=[RecordInputsOutputs()], output_dir=output_dir,
        metadata={'source': evidence, 'source_path': str(Path(source).resolve())}).run()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', required=True)
    parser.add_argument('--report', required=True, help='evaluate.py output of the source run')
    parser.add_argument('--out', required=True)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy('127.0.0.1', args.port)
    try:
        result = run_calls(client, args.record, args.out, report=args.report)
        print(result.value)
        raise SystemExit(0 if result.value['exact'] else 1)
    finally:
        client._ws.close()
