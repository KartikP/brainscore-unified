"""Measure fresh-process native CLI, UMI sessions and recorded UMI experiments."""

from contextlib import contextmanager
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import time

import numpy as np

from brainscore.experiments import Experiment, RecordInputsOutputs, SessionProtocol
from brainscore.models.mirage import MirageRuntime, MirageSubject
from brainscore.models.mirage.model import Clip, OUTPUT_CHANNEL, SUBJECTS
from brainscore_core.streaming import Session, StreamEvent


MODES = ('direct', 'umi', 'recorded')


def execution_orders(repetitions: int) -> list[tuple[str, ...]]:
    """Balance each condition's position across groups of three repetitions."""
    if repetitions < 3 or repetitions % 3:
        raise ValueError('repetitions must be a positive multiple of three')
    forward = [MODES[i:] + MODES[:i] for i in range(3)]
    reverse = [tuple(reversed(order)) for order in forward]
    return list(itertools.islice(itertools.cycle(forward + reverse), repetitions))


def summarize(rows: list[dict]) -> dict:
    measured = [row for row in rows if not row['warmup']]
    summary = {}
    for case in sorted({row['case'] for row in measured}):
        group = [row for row in measured if row['case'] == case]
        modes = {}
        for mode in MODES:
            samples = [row['wall_s'] for row in group if row['mode'] == mode]
            modes[mode] = {'n': len(samples), 'median_s': statistics.median(samples),
                           'min_s': min(samples), 'max_s': max(samples)}
        ratios = {}
        for mode in ('umi', 'recorded'):
            paired = []
            for block in sorted({row['block'] for row in group}):
                values = {row['mode']: row['wall_s'] for row in group if row['block'] == block}
                paired.append(100 * (values[mode] / values['direct'] - 1))
            ratios[mode] = {'paired_overhead_percent': paired,
                            'median_paired_overhead_percent': statistics.median(paired)}
        summary[case] = {'wall': modes, 'relative_to_direct': ratios}
    return summary


def run_process(command: list[str], *, environment: dict, log: Path, timeout_s: float) -> None:
    with log.open('wb') as handle:
        with subprocess.Popen(command, env=environment, stdout=handle, stderr=handle,
                              start_new_session=True) as process:
            try:
                code = process.wait(timeout=timeout_s)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
            if code:
                raise subprocess.CalledProcessError(code, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mirage-python', 'source-dir', 'run-dir', 'work-dir', 'stimuli-dir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--timeout-s', type=float, default=1200)
    args = parser.parse_args()
    orders = execution_orders(args.repetitions)
    args.out.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update({
        'PYTHONPATH': str(args.source_dir.resolve() / 'src'),
        'HF_HOME': str(args.work_dir.resolve() / 'huggingface'),
        'HF_HUB_CACHE': str(args.work_dir.resolve() / 'huggingface/hub'),
        'HF_XET_CACHE': str(args.work_dir.resolve() / 'huggingface/xet'),
        'TORCH_HOME': str(args.work_dir.resolve() / 'torch'),
        'SCRATCHPATH': str(args.work_dir.resolve()),
        'XDG_CACHE_HOME': str(args.work_dir.resolve() / 'xdg'),
        'WANDB_MODE': 'disabled', 'TOKENIZERS_PARALLELISM': 'false',
    })
    rows = []
    qualification = json.loads(Path(__file__).with_name('qualification.json').read_text())
    expected_hashes = dict(zip(('public', 'synthetic'),
                               (case['native_array_sha256'] for case in qualification['cases'])))
    script = Path(__file__).with_name('timing_worker.py').resolve()
    for case, video, transcript, subject_id in (
        ('public', args.stimuli_dir / 'clip.mp4', args.stimuli_dir / 'transcript.json', 'sub-01'),
        ('synthetic', args.stimuli_dir / 'synthetic/clip.mp4',
         args.stimuli_dir / 'synthetic/transcript.json', 'sub-02'),
    ):
        video = video.resolve()
        transcript = transcript.resolve()
        words = json.loads(transcript.read_text())
        reference = None
        for block, order in enumerate([MODES, *orders]):
            for position, mode in enumerate(order):
                trial = args.out / f'{case}-{block:02d}-{position}-{mode}'
                trial.mkdir()
                sidecar = trial / 'worker-timings.json'
                launcher = trial / 'python-launcher'
                launcher.write_text('#!/bin/sh\nexec ' + ' '.join(shlex.quote(str(value)) for value in (
                    args.mirage_python, script, '--timings', sidecar, '--')) + ' "$@"\n')
                launcher.chmod(0o700)
                runtime_s = []
                runtime = MirageRuntime(
                    python_executable=str(launcher), source_dir=args.source_dir,
                    run_dir=args.run_dir, work_dir=args.work_dir, subject=subject_id,
                    device='cuda', batch_size=1, timeout_s=args.timeout_s,
                )

                def predict(clip):
                    start = time.perf_counter()
                    result = runtime(clip)
                    runtime_s.append(time.perf_counter() - start)
                    return result

                @contextmanager
                def factory(_trial):
                    metadata = {'stimulus_id': case}
                    yield Session([
                        StreamEvent('vision:video', video, None, metadata.copy()),
                        StreamEvent('audio', video, None, metadata.copy()),
                        StreamEvent('text', ' '.join(item['word'] for item in words), None,
                                    {**metadata, 'transcript': words}),
                    ], (OUTPUT_CHANNEL,))

                subject = MirageSubject(predictor=predict, subject=subject_id)
                start = time.perf_counter()
                if mode == 'direct':
                    path = trial / 'native.npy'
                    run_process([
                        str(launcher), '-m', 'brain_enc.cli.infer_fmri',
                        '--video', str(video), '--transcript', str(transcript),
                        '--run-dir', str(args.run_dir.resolve()),
                        '--subject-idx', str(SUBJECTS.index(subject_id)),
                        '--device', 'cuda', '--batch-size', '1', '--output', str(path.resolve()),
                    ], environment=environment, log=trial / 'direct.log', timeout_s=args.timeout_s)
                    values = np.load(path, allow_pickle=False)
                elif mode == 'umi':
                    subject.reset()
                    try:
                        with factory(None) as session:
                            subject.participate(session)
                            assembly = session.collect(OUTPUT_CHANNEL)
                        values = assembly.values
                    finally:
                        subject.reset()
                else:
                    result = Experiment(
                        subject=subject,
                        protocol=SessionProtocol(
                            'mirage-overhead', factory, input_channels=('vision:video', 'audio', 'text'),
                            output_channels=(OUTPUT_CHANNEL,),
                        ),
                        tools=(RecordInputsOutputs(),), output_dir=trial / 'experiment',
                        metadata={'case': case, 'block': block, 'position': position,
                                  'purpose': 'timing; no human-data score'},
                    ).run()
                    assembly = result.value[0]
                    values = assembly.values
                wall_s = time.perf_counter() - start
                if hashlib.sha256(values.tobytes()).hexdigest() != expected_hashes[case]:
                    raise AssertionError('Prediction differs from the original GPU qualification')
                if reference is None:
                    reference = values.copy()
                if values.dtype != reference.dtype or not np.array_equal(values, reference):
                    raise AssertionError('Prediction changed during timing qualification')
                if mode != 'direct':
                    assert list(assembly.tr_index.values) == list(range(values.shape[0]))
                    assert list(assembly.parcel_index.values) == list(range(1000))
                    assert set(assembly.subject_id.values) == {subject_id}
                native = json.loads(sidecar.read_text())
                if native['status'] != 'complete':
                    raise RuntimeError('Timing worker did not complete')
                row = {'case': case, 'subject': subject_id, 'block': block, 'position': position,
                       'mode': mode, 'warmup': block == 0, 'wall_s': wall_s,
                       'runtime_call_s': runtime_s[0] if runtime_s else None,
                       'framework_outside_runtime_s': wall_s - runtime_s[0] if runtime_s else None,
                       'native': native, 'shape': list(values.shape), 'dtype': str(values.dtype),
                       'array_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
                       'exact_equal_to_reference': True}
                rows.append(row)
                (trial / 'measurement.json').write_text(json.dumps(row, indent=2) + '\n')
                with (args.out / 'measurements.jsonl').open('a') as log:
                    log.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
    summary = {'conditions': {
        'direct': 'Native CLI plus benchmark-only phase timers',
        'umi': 'Qualified subject with Session, no experiment recorder',
        'recorded': 'Qualified subject in Experiment with RecordInputsOutputs',
    }, 'cache_policy': 'One unmeasured warmup per mode and case; no cache clearing',
        'process_policy': 'Fresh native Python process per call, shared timer launcher',
        'repetitions': args.repetitions, 'results': summarize(rows),
        'scope': 'Both four-second clips, current adapter, single GPU; not persistent-model inference'}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
