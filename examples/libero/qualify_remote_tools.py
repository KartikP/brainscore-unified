"""Compare direct and remote trained-policy calls on identical saved inputs.

Use the complete pinned OpenPI runtime. No simulator or physical robot runs.
"""

import argparse
import json
from pathlib import Path
import threading
import time
import numpy as np


def _check(condition, message):
    # An explicit raise, unlike assert, still runs under python -O.
    if not condition:
        raise RuntimeError(message)


def qualify(policy, requests, out, checkpoint, *, source=None):
    from websockets.sync.server import serve
    from brainscore.experiments import (
        OpenPIInstrumentation,
        OpenPIToolServer,
        OpenPIPolicyClient,
        RemoteOpenPIInstrumentation,
        Experiment,
        CallableProtocol,
        RecordInputsOutputs,
        RecordActivity,
        Ablate,
    )
    from brainscore_core.events import Selection
    from brainscore.run_record import RunRecord

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rng = policy._rng
    baseline = []
    baseline_times = []
    for request in requests:
        start = time.monotonic()
        baseline.append(policy.infer(request)['actions'].copy())
        baseline_times.append(time.monotonic() - start)
    policy._rng = rng
    backend = OpenPIInstrumentation(policy, checkpoint=checkpoint, steps=[0, 1])
    server = OpenPIToolServer(policy, backend)
    target = Selection(layer='action_out_proj', indices=[0])
    report = {
        'scope': 'trained same-input remote policy; no simulator',
        'calls': len(requests),
        'checkpoint': checkpoint,
        'source': source,
        'instrumentation': backend.describe(),
        'baseline_seconds': baseline_times,
    }
    try:
        with serve(
            server.handler, '127.0.0.1', 0, compression=None, max_size=64 * 1024 * 1024
        ) as ws:
            thread = threading.Thread(target=ws.serve_forever, daemon=True)
            thread.start()
            uri = f'ws://127.0.0.1:{ws.socket.getsockname()[1]}'
            try:
                for name, tools in [
                    ('recording', [RecordActivity([target])]),
                    (
                        'ablation',
                        [
                            RecordActivity([target], when='before', name='before'),
                            Ablate([target]),
                            RecordActivity([target], name='after'),
                        ],
                    ),
                    ('restored', []),
                ]:
                    policy._rng = rng
                    outputs = []
                    timings = []

                    def evaluate(subject, context):
                        for request in requests:
                            start = time.monotonic()
                            outputs.append(subject.infer(request)['actions'].copy())
                            timings.append(time.monotonic() - start)
                        return {'calls': len(outputs)}

                    with OpenPIPolicyClient(uri) as client:
                        Experiment(
                            subject=client,
                            protocol=CallableProtocol(
                                name, evaluate, methods=['infer']
                            ),
                            tools=[RecordInputsOutputs(), *tools],
                            instrumentation=RemoteOpenPIInstrumentation(client),
                            output_dir=out / name,
                        ).run()
                    report[name + '_seconds'] = timings
                    equal = len(outputs) == len(baseline) and all(
                        np.array_equal(a, b) for a, b in zip(baseline, outputs))
                    report[name + '_exact_baseline'] = equal
                    if name != 'ablation':
                        _check(equal, f'{name} actions differ from direct baseline')
                    else:
                        # Informative only: an intervention need not change actions.
                        report['effects'] = {'ablation_changed_actions': not equal}
                        _check(len(outputs) == len(baseline)
                               and all(np.isfinite(a).all() for a in outputs),
                               'Ablation outputs are missing or nonfinite')
                        activity = [e for e in RunRecord(out / name).events()
                                    if e['kind'] == 'activity']
                        after = [e['payload']['value']['array'] for e in activity
                                 if e['source'] == 'after']
                        before = [e['payload']['value']['array'] for e in activity
                                  if e['source'] == 'before']
                        _check(after and all(np.all(v == 0) for v in after),
                               'Selected unit not silenced')
                        _check(any(np.any(v != 0) for v in before),
                               'Selected unit was already zero before ablation')
                        report['selected_unit_zeroed'] = True
                        report['selected_unit_was_nonzero'] = True
                report['status'] = 'passed'
            finally:
                ws.shutdown()
                thread.join(timeout=10)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        policy._rng = rng
        (out / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path, help='evaluate.py output of the source run')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    from itertools import islice
    from brainscore.run_record import RunRecord
    from replay import pairs, verify_capture
    from serve_tools import checkpoint_digest, load_policy

    record = RunRecord(args.record)
    source = verify_capture(record, args.report)
    requests = [request for request, _ in islice(pairs(record, source['calls']), 10)]
    report = qualify(
        load_policy('pi05_libero', args.checkpoint),
        requests,
        args.out,
        'pi05_libero:sha256:' + checkpoint_digest(args.checkpoint),
        source=source,
    )
    print(json.dumps(report, indent=2))
