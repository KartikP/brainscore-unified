"""Summarize complete paired evaluations without asserting scientific equivalence."""
import argparse
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


def compare(reference_dir, umi_dir):
    reports, trials = [], []
    for directory in (Path(reference_dir), Path(umi_dir)):
        report = json.loads((directory / 'report.json').read_text())
        rows = [json.loads(line) for line in (directory / 'trials.jsonl').read_text().splitlines()]
        indexed = {(row['task'], row['trial']): row for row in rows}
        if (report['status'] != 'complete' or report['errors'] or not rows
                or len(rows) != report['expected_episodes'] or len(indexed) != len(rows)
                or any(row['errors'] for row in rows)
                or sum(row['success'] for row in rows) != report['successes']):
            raise ValueError('Incomplete, inconsistent, or failed evaluation')
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
    result = {'episodes_per_route': n, 'suite': reports[0]['suite'],
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
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.umi)
    with Path(args.out).open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
