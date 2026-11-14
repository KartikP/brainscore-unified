"""Compare direct released MIRAGE inference with UMI recording and input replay."""

from contextlib import contextmanager
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from brainscore.experiments import (
    Experiment, RecordInputsOutputs, SessionProtocol, compare_outputs, replay_sessions,
)
from brainscore.models.mirage import MirageRuntime, MirageSubject
from brainscore.models.mirage.model import OUTPUT_CHANNEL, PARTICIPANTS
from brainscore_core.streaming import Session, StreamEvent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mirage-python', required=True)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--transcript', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--participant', default='sub-01', choices=PARTICIPANTS)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--timeout-s', type=float, default=1200.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    words = json.loads(args.transcript.read_text())
    local_video = args.out / 'stimulus.mp4'
    shutil.copyfile(args.video, local_video)
    direct_path = args.out / 'direct.npy'
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
    command = [
        args.mirage_python, '-m', 'brain_enc.cli.infer_fmri',
        '--video', str(local_video.resolve()),
        '--transcript', str(args.transcript.resolve()),
        '--run-dir', str(args.run_dir.resolve()),
        '--subject-idx', str(PARTICIPANTS.index(args.participant)),
        '--device', args.device, '--batch-size', '1', '--output', str(direct_path.resolve()),
    ]
    with (args.out / 'direct.log').open('wb') as log:
        subprocess.run(command, env=environment, stdout=log, stderr=log,
                       timeout=args.timeout_s, check=True)
    native = np.load(direct_path, allow_pickle=False)
    runtime = MirageRuntime(
        python_executable=args.mirage_python, source_dir=args.source_dir,
        run_dir=args.run_dir, work_dir=args.work_dir, participant=args.participant,
        device=args.device, batch_size=1, timeout_s=args.timeout_s,
    )
    subject = MirageSubject(predictor=runtime, participant=args.participant)

    @contextmanager
    def factory(_trial):
        metadata = {'stimulus_id': 'demo-clip'}
        yield Session([
            StreamEvent('vision:video', local_video.resolve(), None, metadata.copy()),
            StreamEvent('audio', local_video.resolve(), None, metadata.copy()),
            StreamEvent('text', ' '.join(word['word'] for word in words), None,
                        {**metadata, 'transcript': words}),
        ], (OUTPUT_CHANNEL,))

    result = Experiment(
        subject=subject,
        protocol=SessionProtocol(
            'mirage-native-fidelity', factory,
            required_input_channels=('vision:video', 'audio', 'text'), requested_output_channels=(OUTPUT_CHANNEL,),
        ),
        tools=(RecordInputsOutputs(),), output_dir=args.out / 'record',
        metadata={'purpose': 'integration fidelity; no human response data or fitting'},
    ).run()
    assembly = result.value[0]
    if not np.array_equal(native, assembly.values) or native.dtype != assembly.dtype:
        raise AssertionError('Direct native and UMI predictions differ; investigate without a tolerance')
    if list(assembly.parcel_index.values) != list(range(1000)):
        raise AssertionError('Parcel ordering changed')
    if list(assembly.tr_index.values) != list(range(native.shape[0])):
        raise AssertionError('TR ordering changed')
    local_video.unlink()
    replay = Experiment(
        subject=subject, protocol=replay_sessions(result.directory),
        tools=(RecordInputsOutputs(),), output_dir=args.out / 'replay',
        metadata={'mode': 'saved inputs; rerun native model after removing original copy'},
    ).run()
    comparison = compare_outputs(result.directory, replay.directory)
    if not comparison['equal']:
        raise AssertionError('Input replay changed the native outputs')
    summary = {
        'direct_equals_umi_exactly': True, 'replay_equals_original_exactly': True,
        'shape': list(native.shape), 'dtype': str(native.dtype),
        'native_array_sha256': hashlib.sha256(native.tobytes()).hexdigest(),
        'parcel_order': 'native zero-based indices 0..999',
        'tr_order': 'native zero-based output row order',
        'signal': 'predicted BOLD; no additional readout or HRF applied',
        'provenance': assembly.attrs['provenance'],
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
