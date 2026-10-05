"""Compare recorded actions with fresh predictions for the same observations."""
import argparse
import json
from pathlib import Path

import numpy as np

from brainscore.run_record import RunRecord


def compare_record(directory, policy, *, atol=0.0):
    if not np.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and nonnegative')
    record = RunRecord(directory)
    if record.manifest.get('status') != 'complete':
        raise ValueError('Expected a completed record')
    differences = []
    events = iter(record.events())
    for input_event in events:
        output_event = next(events, None)
        if (input_event['kind'] != 'input' or output_event is None
                or output_event['kind'] != 'output'):
            raise ValueError('Expected alternating input/output pairs')
        request, expected = input_event['payload'], output_event['payload']
        actual = np.asarray(policy.infer(request)['actions'])
        target = np.asarray(expected['actions'])
        if actual.shape != target.shape or not np.isfinite(actual).all() or not np.isfinite(target).all():
            raise ValueError('Action shape changed or actions contain nonfinite values')
        differences.append(float(np.max(np.abs(actual.astype(float) - target.astype(float)))))
    if not differences:
        raise ValueError('Expected a nonempty record')
    return {'calls': len(differences), 'atol': atol, 'rtol': 0,
            'source_run_id': record.manifest['run_id'],
            'source_events_sha256': record.manifest.get('events_sha256'),
            'max_absolute_error': max(differences),
            'failed_calls': sum(delta > atol for delta in differences),
            'passed': all(delta <= atol for delta in differences),
            'scope': 'same recorded observations; not closed-loop task success'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', required=True)
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--out', required=True)
    parser.add_argument('--atol', type=float, default=0.0)
    args = parser.parse_args()
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    report = compare_record(args.record, WebsocketClientPolicy('127.0.0.1', args.port), atol=args.atol)
    with Path(args.out).open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
