"""Summarize complete paired evaluations without asserting scientific equivalence.

Each side is labelled with its scope. Reports from serve_bridge.py are
'transport' only. A report from serve_tool_bridge.py also needs that run's
UMI experiment directory; its records, receipt and acknowledgment are verified
read-only with RunRecord, so that check runs in the UMI (Python 3.11)
environment. Nothing is written to either run directory.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - half), min(1, center + half)]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_tool_run(directory, report, rows, experiment, *, reconcile_unacknowledged=False):
    """Join the evaluator report to its UMI record; return the qualified scope."""
    from brainscore.run_record import RunRecord  # UMI environment only

    directory, experiment = Path(directory), Path(experiment)
    run_id = report['bridge']['umi_run_id']
    manifest = json.loads((experiment / 'experiment.json').read_text())
    _check(manifest['status'] == 'complete', 'UMI experiment did not complete')
    _check(manifest['run_id'] == run_id, 'UMI experiment belongs to another run')
    for artifact in manifest['artifacts']:
        _check(_sha256(experiment / artifact['path']) == artifact['sha256'],
               'Artifact changed after the run: ' + artifact['path'])
    _check('result.json' in [artifact['path'] for artifact in manifest['artifacts']],
           'UMI experiment has no recorded result')
    record = RunRecord(experiment)
    _check(record.manifest['status'] == 'complete', 'UMI call record did not complete')
    result = record.codec.decode(json.loads((experiment / 'result.json').read_text()))
    report_sha256 = _sha256(directory / 'report.json')
    _check(isinstance(result, dict) and set(result) == {'calls', 'report_sha256', 'run_id'}
           and result['report_sha256'] == report_sha256 and result['run_id'] == run_id,
           'UMI result does not match this report')
    receipts, annotations = [], []
    for event in record.events():  # checksum-verified read of every event
        if event['kind'] == 'evaluator_receipt':
            receipts.append(event['payload'])
        elif event['kind'] == 'input' and event['payload'].get('method') == 'process':
            annotations.append(event['payload']['args'][0].context['umi_call'])
    _check(len(receipts) == 1, 'Expected exactly one recorded evaluator receipt')
    receipt = receipts[0]
    _check(receipt['report_sha256'] == report_sha256 and receipt['run_id'] == run_id,
           'Recorded receipt does not match this report')
    _check([a['index'] for a in annotations] == list(range(result['calls'])),
           'Recorded calls are missing or out of order')
    _check(len(receipt['episodes']) == len(rows)
           and sum(row['calls'] for row in rows) == len(annotations), 'Receipt and trials differ')
    for episode, row in zip(receipt['episodes'], rows):
        expected = {key: row.get(key) for key in episode}
        expected['errors'] = len(row['errors'])
        _check(episode == expected, 'Receipt and trials differ for ' + str((row['task'], row['trial'])))
        calls = sum(a['task'] == row['task'] and a['trial'] == row['trial'] for a in annotations)
        _check(calls == row['calls'], 'Recorded calls differ for ' + str((row['task'], row['trial'])))
    finish_path = directory / 'finish.json'
    finish = json.loads(finish_path.read_text()) if finish_path.is_file() else {}
    if (finish.get('ack') == 'complete' and finish.get('run_id') == run_id
            and finish.get('manifest_sha256') == _sha256(experiment / 'experiment.json')):
        return 'tool-bridge'
    _check(reconcile_unacknowledged, 'Bridge completion was not acknowledged; '
                                     'use --reconcile-unacknowledged to check the final records')
    return 'tool-bridge-reconciled-unacknowledged'


def compare(reference_dir, umi_dir, *, reference_experiment=None, umi_experiment=None,
            reconcile_unacknowledged=False):
    reports, trials, scopes = [], [], {}
    sides = (('reference', reference_dir, reference_experiment), ('umi', umi_dir, umi_experiment))
    for label, directory, experiment in sides:
        directory = Path(directory)
        report = json.loads((directory / 'report.json').read_text())
        rows = [json.loads(line) for line in (directory / 'trials.jsonl').read_text().splitlines()]
        indexed = {(row['task'], row['trial']): row for row in rows}
        if (report['status'] != 'complete' or report['errors'] or not rows
                or len(rows) != report['expected_episodes'] or len(indexed) != len(rows)
                or any(row['errors'] or row.get('ended_by') == 'error' for row in rows)
                or sum(row['success'] for row in rows) != report['successes']):
            raise ValueError('Incomplete, inconsistent, or failed evaluation')
        if (report.get('bridge') or {}).get('receipt_protocol'):
            if experiment is None:
                raise ValueError(label + ' used the tool bridge; pass its UMI experiment directory')
            scopes[label] = verify_tool_run(directory, report, rows, experiment,
                                            reconcile_unacknowledged=reconcile_unacknowledged)
        elif experiment is not None:
            raise ValueError(label + ' is a transport-only report; it has no UMI experiment to verify')
        else:
            scopes[label] = 'transport'
        reports.append(report)
        trials.append(indexed)
    for key in ('suite', 'trials_per_task', 'openpi_revision', 'libero_revision', 'evaluator_sha256'):
        if reports[0][key] != reports[1][key]:
            raise ValueError('Protocol mismatch: ' + key)
    # Route, port and output paths differ; all scientific settings must agree.
    settings = [{k: v for k, v in report['settings'].items()
                 if k not in ('host', 'port', 'video_out_path')} for report in reports]
    if settings[0] != settings[1] or trials[0].keys() != trials[1].keys():
        raise ValueError('Unmatched trials or settings')
    reference_only = umi_only = 0
    for key in trials[0]:
        first, second = trials[0][key], trials[1][key]
        if first['initial_state_sha256'] != second['initial_state_sha256']:
            raise ValueError('Initial state mismatch: ' + str(key))
        reference_only += bool(first['success'] and not second['success'])
        umi_only += bool(second['success'] and not first['success'])
    n = len(trials[0])
    result = {'episodes_per_route': n, 'suite': reports[0]['suite'], 'scope': scopes,
              'reference_only_successes': reference_only, 'umi_only_successes': umi_only,
              'success_rate_difference': (umi_only - reference_only) / n,
              'interpretation': 'descriptive comparison; not proof of equivalence',
              'interval_note': '95% Wilson interval pooled across trials; task dependence is not modeled'}
    for label, report in zip(('reference', 'umi'), reports):
        result[label] = {'successes': report['successes'], 'success_rate': report['successes'] / n,
                         'wilson_95': wilson(report['successes'], n)}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--umi', required=True)
    parser.add_argument('--reference-experiment', help='UMI experiment directory of a tool-bridge reference run')
    parser.add_argument('--umi-experiment', help='UMI experiment directory of a tool-bridge run')
    parser.add_argument('--reconcile-unacknowledged', action='store_true',
                        help='Accept a completed bridge record whose acknowledgment was lost')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.umi, reference_experiment=args.reference_experiment,
                     umi_experiment=args.umi_experiment,
                     reconcile_unacknowledged=args.reconcile_unacknowledged)
    with Path(args.out).open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
