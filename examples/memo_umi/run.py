"""Run the paper's three-object RSA task natively and through UMI."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path

import numpy as np

from brainscore.experiments import (
    Experiment, RecordInputsOutputs, SessionProtocol, Trial,
    compare_outputs, read_events, replay_sessions,
)
from brainscore_core.streaming import BufferedSession, StreamEvent

from brainscore.memo_umi import MemoConfig, MemoSubject, MemoWorker
from brainscore.memo_umi.subject import CHOICES, SCENE, UTTERANCES


@contextmanager
def session_factory(_trial: Trial):
    events = [StreamEvent('instruction', SCENE, 0.0)]
    events.extend(
        StreamEvent('text', utterance, float(index + 1), {'stimulus_id': utterance})
        for index, utterance in enumerate(UTTERANCES)
    )
    yield BufferedSession(events, requested_output_channels=('behavior',))


def protocol():
    return SessionProtocol(
        'memo-rsa-green-pink-square-round', session_factory,
        trials=('all-four-utterances',),
        input_channels=('instruction', 'text'), output_channels=('behavior',),
        metadata={'utterance_order': UTTERANCES, 'choice_order': CHOICES,
                  'scene': SCENE},
    )


def run_depth(*, depth, beta, python_executable, out_dir, replay=False):
    subject = MemoSubject(
        config=MemoConfig(depth=depth, beta=beta),
        python_executable=python_executable,
    )
    if replay:
        source = out_dir / f'depth-{depth}'
        return Experiment(
            subject=subject, protocol=replay_sessions(source),
            tools=(RecordInputsOutputs(),), output_dir=out_dir / f'depth-{depth}-replay',
            metadata={'purpose': 'replay_saved_inputs_rerunning_memo',
                      'source_record': str(source)},
        ).run()
    return Experiment(
        subject=subject, protocol=protocol(), tools=(RecordInputsOutputs(),),
        output_dir=out_dir / f'depth-{depth}',
        metadata={'comparison': 'model_configuration_only', 'reasoning_depth': depth,
                  'beta': beta, 'python_executable': python_executable},
    ).run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memo-python', default=os.environ.get('MEMO_PYTHON'),
                        help='Python >=3.12 environment with memo-lang and CPU JAX')
    parser.add_argument('--out', type=Path, default=Path('memo-umi-output'))
    parser.add_argument('--beta', type=float, default=1.0)
    args = parser.parse_args()
    if not args.memo_python:
        parser.error('pass --memo-python or set MEMO_PYTHON')
    args.out.mkdir(parents=True, exist_ok=False)

    summary = {'task': 'Lewis three-object speaker-listener RSA', 'beta': args.beta,
               'choice_order': CHOICES, 'utterance_order': UTTERANCES,
               'comparison': 'model configuration only; no human data fit'}
    depth_inputs = []
    for depth in (0, 1):
        native = MemoWorker(args.memo_python)
        native_rows = native.predict(depth=depth, beta=args.beta,
                                     utterances=list(UTTERANCES))
        native_provenance = native.runtime
        native.close()
        recorded = run_depth(depth=depth, beta=args.beta,
                             python_executable=args.memo_python, out_dir=args.out)
        depth_inputs.append([
            event['payload'] for event in read_events(recorded.directory)
            if event['kind'] == 'input'
        ])
        behavior = recorded.value[0]
        if not np.array_equal(np.asarray(native_rows, dtype=np.float64), behavior.values):
            raise AssertionError(f'direct memo and UMI tables differ at depth={depth}')
        table = {
            utterance: {choice: float(probability)
                        for choice, probability in zip(CHOICES, row)}
            for utterance, row in zip(UTTERANCES, native_rows)
        }
        replayed = run_depth(depth=depth, beta=args.beta,
                             python_executable=args.memo_python, out_dir=args.out,
                             replay=True)
        exact_replay = compare_outputs(recorded.directory, replayed.directory)
        if not exact_replay['equal']:
            raise AssertionError(f'input replay changed memo outputs at depth={depth}')
        summary[f'depth_{depth}'] = {
            'native_probabilities': table,
            'umi_probabilities': table,
            'native_runtime': native_provenance,
            'record': str(recorded.directory),
            'replay': str(replayed.directory),
            'replay_mode': 'saved inputs; memo rerun; exact outputs verified',
        }
        print(f'depth={depth}: direct memo == UMI exactly; replay == original exactly')

    if depth_inputs[0] != depth_inputs[1]:
        raise AssertionError('reasoning-depth conditions did not receive identical inputs')
    summary['input_comparison'] = 'identical recorded events at both depths'

    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
