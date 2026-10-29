"""Compare direct, recorded, ablated, and restored OpenPI inference.

Use saved requests from a serve_bridge.py capture whose evaluator report is
complete (replay.verify_capture) and a local checkpoint. This tests identical
inputs; it does not run physics or measure task success.
"""
import argparse
import json
from pathlib import Path

import numpy as np


# Qualification gates. Exact action parity is required; a changed action is reported, not gated.
GATES = ('recording_preserves_actions', 'cleanup_restores_actions', 'selected_activity_zeroed',
         'selected_activity_was_nonzero', 'ablation_outputs_finite')


def qualify(policy, requests, output_dir, *, checkpoint, steps=None):
    from brainscore.experiments import (
        Experiment, CallableProtocol, RecordInputsOutputs, RecordActivity,
        Ablate, OpenPIInstrumentation,
    )
    from brainscore.run_record import RunRecord
    from brainscore_core.events import Selection

    if not requests:
        raise ValueError('Provide at least one recorded request')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    instrumentation = OpenPIInstrumentation(policy, checkpoint=checkpoint, steps=steps)
    target = Selection(layer='action_out_proj', indices=[0])
    initial_rng = policy._rng
    report = {'status': 'running', 'scope': 'same-input policy inference; no simulator',
              'calls_per_route': len(requests), 'atol': 0, 'rtol': 0,
              'checkpoint': checkpoint, 'instrumentation': instrumentation.describe()}

    def run(name, tools):
        policy._rng = initial_rng
        outputs = []
        def evaluate(subject, context):
            for request in requests:
                outputs.append(subject.infer(request)['actions'].copy())
            return {'calls': len(outputs)}
        Experiment(
            subject=policy,
            protocol=CallableProtocol(name, evaluate, methods=['infer']),
            tools=[RecordInputsOutputs(), *tools],
            instrumentation=instrumentation,
            output_dir=output_dir / name,
        ).run()
        return outputs

    try:
        baseline = run('baseline', [])
        recorded = run('recording', [RecordActivity([target])])
        ablated = run('ablation', [
            RecordActivity([target], when='before', name='before'),
            Ablate([target]),
            RecordActivity([target], name='after'),
        ])
        restored = run('restored', [])
        def exact(left, right):
            return len(left) == len(right) == len(requests) and all(
                np.array_equal(a, b) and np.isfinite(a).all() and np.isfinite(b).all()
                for a, b in zip(left, right))
        report['recording_preserves_actions'] = exact(baseline, recorded)
        report['cleanup_restores_actions'] = exact(baseline, restored)
        activity = [row['payload'] for row in RunRecord(output_dir / 'ablation' / 'inputs_outputs').events()
                    if row['payload']['kind'] == 'activity']
        after = [event['payload']['value']['array'] for event in activity if event['source'] == 'after']
        before = [event['payload']['value']['array'] for event in activity if event['source'] == 'before']
        report['selected_activity_zeroed'] = bool(after) and all(np.all(value == 0) for value in after)
        report['selected_activity_was_nonzero'] = any(np.any(value != 0) for value in before)
        report['ablation_outputs_finite'] = (len(ablated) == len(requests)
                                             and all(np.isfinite(value).all() for value in ablated))
        # Informative only: an intervention need not change actions or task outcomes.
        report['effects'] = {'intervention_changed_actions': not exact(baseline, ablated)}
        report['status'] = 'passed' if all(report[gate] for gate in GATES) else 'failed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        policy._rng = initial_rng
        (output_dir / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path, help='evaluate.py output of the source run')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', default='pi05_libero')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--steps', nargs='+', type=int)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    from itertools import islice
    from brainscore.run_record import RunRecord
    from replay import pairs, verify_capture
    from serve_tools import checkpoint_digest, load_policy
    record = RunRecord(args.record)
    source = verify_capture(record, args.report)
    requests = [request for request, _ in islice(pairs(record, source['calls']), args.limit)]
    checkpoint = args.checkpoint.resolve(strict=True)
    identity = f'{args.config}:sha256:{checkpoint_digest(checkpoint)}'
    report = qualify(
        load_policy(args.config, checkpoint), requests, args.out,
        checkpoint=identity, steps=args.steps,
    )
    report['source'] = source
    (args.out / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
