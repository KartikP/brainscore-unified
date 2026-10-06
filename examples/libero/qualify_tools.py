"""Compare direct, recorded, ablated, and restored OpenPI inference.

Use saved requests from a completed LIBERO call record and a local checkpoint.
This tests identical inputs; it does not run physics or measure task success.
"""
import argparse
import json
from pathlib import Path

import numpy as np


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
              'instrumentation': instrumentation.describe()}

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
            return all(np.array_equal(a, b) and np.isfinite(a).all() and np.isfinite(b).all()
                       for a, b in zip(left, right))
        report['recording_preserves_actions'] = exact(baseline, recorded)
        report['cleanup_restores_actions'] = exact(baseline, restored)
        report['intervention_changes_actions'] = not exact(baseline, ablated)
        activity = [row['payload'] for row in RunRecord(output_dir / 'ablation' / 'inputs_outputs').events()
                    if row['payload']['kind'] == 'activity']
        after = [event['payload']['value']['array'] for event in activity if event['source'] == 'after']
        before = [event['payload']['value']['array'] for event in activity if event['source'] == 'before']
        report['selected_activity_zeroed'] = bool(after) and all(np.all(value == 0) for value in after)
        report['selected_activity_was_nonzero'] = any(np.any(value != 0) for value in before)
        report['ablation_outputs_finite'] = all(np.isfinite(value).all() for value in ablated)
        checks = [value for value in report.values() if isinstance(value, bool)]
        report['status'] = 'passed' if all(checks) else 'failed'
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
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', default='pi05_libero')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--steps', nargs='+', type=int)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    from brainscore.run_record import RunRecord
    from serve_tools import checkpoint_digest, load_policy
    record = RunRecord(args.record)
    if record.manifest['status'] != 'complete':
        parser.error('Use a completed source record')
    events = list(record.events())
    requests = [event['payload'] for event in events if event['kind'] == 'input'][:args.limit]
    checkpoint = args.checkpoint.resolve(strict=True)
    identity = f'{args.config}:sha256:{checkpoint_digest(checkpoint)}'
    report = qualify(
        load_policy(args.config, checkpoint), requests, args.out,
        checkpoint=identity, steps=args.steps,
    )
    report['source_events_sha256'] = record.manifest['events_sha256']
    (args.out / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
