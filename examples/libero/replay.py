"""Compare recorded actions with fresh predictions for the same observations.

A serve_bridge.py record is a transport capture: 'complete' means its one
evaluator session closed with every request answered, not that the evaluation
finished. Before any record qualifies a replay or tool comparison,
verify_capture requires that run's completed evaluator report and trials:
pinned revisions and settings, every declared episode ending by success or the
step limit (checked with the tool bridge's own receipt rules), and exactly two
recorded events per reported call, read in strict request/reply alternation
through the whole log (bounded memory). The transport route has no run identifier
shared by the report and the record, so the pairing rests on these checks and
on keeping each run's outputs together; it is not cryptographic proof.
"""
import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from brainscore.run_record import RunRecord


@lru_cache(maxsize=None)
def _sibling(name):
    path = Path(__file__).resolve().with_name(name + '.py')
    spec = importlib.util.spec_from_file_location('libero_replay_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_capture(record, report_dir, *, suite=None, trials_per_task=None):
    """Return the evaluator evidence for a complete transport capture, or raise ValueError.

    Every request/reply pair is checked before returning, so callers may then
    take only a prefix of pairs(record, evidence['calls']).
    """
    bridge, evaluate, rules = _sibling('serve_bridge'), _sibling('evaluate'), _sibling('serve_tool_bridge')
    _check(record.manifest.get('status') == 'complete', 'Expected a completed record')
    metadata = record.metadata
    _check(isinstance(metadata, dict) and metadata.get('capture') == bridge.CAPTURE,
           'Not a serve_bridge.py transport capture; older captures cannot detect interrupted calls')
    report_dir = Path(report_dir)
    data = (report_dir / 'report.json').read_bytes()
    try:
        report = json.loads(data)
        rows = [json.loads(line) for line in (report_dir / 'trials.jsonl').read_text().splitlines()]
        _check(report['status'] == 'complete' and report['bridge'] is None
               and report['route'] == metadata['route'],
               'Report is not a completed transport evaluation of this route')
        _check(not report['errors'] and not report['cleanup_errors'], 'Evaluator reported errors')
        _check(report['openpi_revision'] == evaluate.OPENPI_REVISION
               and report['libero_revision'] == evaluate.LIBERO_REVISION,
               'Report is not from the pinned evaluator')
        name, trials, horizon = report['suite'], report['trials_per_task'], report['execution_horizon']
        _check(suite in (None, name) and trials_per_task in (None, trials),
               'Report suite or trials differ from the requested evaluation')
        settings = report['settings']
        _check(name in rules.SUITES and report['max_steps'] == evaluate.MAX_STEPS[name]
               and settings['task_suite_name'] == name and settings['num_trials_per_task'] == trials
               and settings['replan_steps'] == horizon
               and settings['num_steps_wait'] == report['num_steps_wait'],
               'Report settings are inconsistent with the pinned evaluator')
        start = {'version': rules.RECEIPT_PROTOCOL, 'run_id': 'transport', 'suite': name,
                 'trials_per_task': trials, 'execution_horizon': horizon,
                 'num_steps_wait': report['num_steps_wait'], 'max_steps': report['max_steps'],
                 'tasks': list(dict.fromkeys(row['task'] for row in rows))}
        keys = rules.check_start(start, run_id='transport', suite=name, trials_per_task=trials,
                                 horizon=horizon)
        episodes = [dict({key: row[key] for key in rules.EPISODE_FIELDS - {'errors'}},
                         errors=len(row['errors'])) for row in rows]
        receipt = {'version': rules.RECEIPT_PROTOCOL, 'run_id': 'transport',
                   'report_sha256': hashlib.sha256(data).hexdigest(), 'status': report['status'],
                   'episodes': episodes}
        # The bridge's own call counts do not exist here; the record total is checked below.
        rules.check_receipt(receipt, run_id='transport', start=start, keys=keys,
                            calls=[row['calls'] for row in episodes], horizon=horizon)
        _check(report['episodes'] == report['expected_episodes'] == len(keys)
               and report['successes'] == sum(row['success'] for row in rows),
               'Report totals differ from its trials')
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
        raise ValueError('Malformed evaluator report or trials') from error
    calls = sum(row['calls'] for row in rows)
    _check(record.manifest.get('event_count') == 2 * calls,
           f"Record has {record.manifest.get('event_count')} events; the report accounts for "
           f'{calls} calls (two events each)')
    # Stream the whole log once, so callers may then take only its first requests.
    for _ in pairs(record, calls):
        pass
    return {'calls': calls, 'route': metadata['route'], 'suite': name, 'trials_per_task': trials,
            'episodes': len(rows), 'report_sha256': receipt['report_sha256'],
            'source_run_id': record.manifest['run_id'],
            'source_events_sha256': record.manifest['events_sha256']}


def pairs(record, calls):
    """Yield (request, output) payloads; strict input/output alternation, `calls` pairs."""
    events, count = iter(record.events()), 0
    for event in events:
        output = next(events, None)
        _check(event['kind'] == 'input' and output is not None and output['kind'] == 'output',
               'Expected alternating input/output pairs')
        count += 1
        yield event['payload'], output['payload']
    _check(count == calls, f'Record has {count} pairs; expected {calls}')


def compare_record(directory, policy, *, report, atol=0.0, suite=None, trials_per_task=None):
    if not np.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and nonnegative')
    record = RunRecord(directory)
    evidence = verify_capture(record, report, suite=suite, trials_per_task=trials_per_task)
    differences = []
    for request, expected in pairs(record, evidence['calls']):
        actual = np.asarray(policy.infer(request)['actions'])
        target = np.asarray(expected['actions'])
        if actual.shape != target.shape or not np.isfinite(actual).all() or not np.isfinite(target).all():
            raise ValueError('Action shape changed or actions contain nonfinite values')
        differences.append(float(np.max(np.abs(actual.astype(float) - target.astype(float)))))
    return {'calls': len(differences), 'atol': atol, 'rtol': 0,
            'source': evidence,
            'source_run_id': evidence['source_run_id'],
            'source_events_sha256': evidence['source_events_sha256'],
            'max_absolute_error': max(differences),
            'failed_calls': sum(delta > atol for delta in differences),
            'passed': all(delta <= atol for delta in differences),
            'scope': 'same recorded observations; not closed-loop task success'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', required=True)
    parser.add_argument('--report', required=True,
                        help="evaluate.py output directory of the run that produced --record")
    parser.add_argument('--suite')
    parser.add_argument('--trials-per-task', type=int)
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--out', required=True)
    parser.add_argument('--atol', type=float, default=0.0)
    args = parser.parse_args()
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    report = compare_record(args.record, WebsocketClientPolicy('127.0.0.1', args.port),
                            report=args.report, atol=args.atol, suite=args.suite,
                            trials_per_task=args.trials_per_task)
    with Path(args.out).open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
