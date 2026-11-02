"""A bounded subprocess around the unmodified upstream raw-video pipeline."""

from __future__ import annotations

import json
import math
import os
import signal
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .model import Clip, Prediction, SUBJECTS


class MirageRuntimeError(RuntimeError):
    """Native MIRAGE inference failed."""


class MirageTimeout(MirageRuntimeError):
    """The entire native worker exceeded its deadline."""


class MirageRuntime:
    """Run one clip in an isolated Python 3.12 process; retain no GPU state."""

    def __init__(
        self,
        *,
        python_executable: str,
        source_dir: Path,
        run_dir: Path,
        work_dir: Path,
        subject: str = 'sub-01',
        device: str = 'cuda',
        batch_size: int = 1,
        timeout_s: float = 1200.0,
    ) -> None:
        if subject not in SUBJECTS:
            raise ValueError(f'subject must be one of {SUBJECTS}')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('timeout_s must be finite and positive')
        self.python_executable = python_executable
        self.source_dir = Path(source_dir).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.subject = subject
        self.device = device
        self.batch_size = batch_size
        self.timeout_s = timeout_s

    def __call__(self, clip: Clip) -> Prediction:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='mirage-', dir=self.work_dir) as directory:
            root = Path(directory)
            # Recorded Path assets have checksum names without their media suffix.
            video = root / 'video.mp4'
            video.symlink_to(clip.video.resolve())
            transcript = root / 'transcript.json'
            transcript.write_text(json.dumps(clip.transcript, allow_nan=False))
            request = {
                'video': str(video), 'transcript': str(transcript),
                'run_dir': str(self.run_dir), 'subject': self.subject,
                'device': self.device, 'batch_size': self.batch_size,
                'output': str(root / 'prediction.npy'),
                'provenance': str(root / 'provenance.json'),
            }
            request_path = root / 'request.json'
            request_path.write_text(json.dumps(request, allow_nan=False))
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(self.source_dir / 'src')
            environment['HF_HOME'] = str(self.work_dir / 'huggingface')
            environment['HF_HUB_CACHE'] = str(self.work_dir / 'huggingface/hub')
            environment['HF_XET_CACHE'] = str(self.work_dir / 'huggingface/xet')
            environment['TORCH_HOME'] = str(self.work_dir / 'torch')
            environment['SCRATCHPATH'] = str(self.work_dir)
            environment['XDG_CACHE_HOME'] = str(self.work_dir / 'xdg')
            environment['WANDB_MODE'] = 'disabled'
            environment['TOKENIZERS_PARALLELISM'] = 'false'
            command = [self.python_executable, str(Path(__file__).with_name('worker.py')),
                       str(request_path)]
            try:
                with (root / 'stdout.log').open('wb') as stdout, (root / 'stderr.log').open('wb') as stderr:
                    with subprocess.Popen(
                        command, env=environment, stdout=stdout, stderr=stderr,
                        start_new_session=True,
                    ) as process:
                        try:
                            returncode = process.wait(timeout=self.timeout_s)
                        except BaseException:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait()
                            raise
                        if returncode:
                            raise subprocess.CalledProcessError(returncode, command)
            except subprocess.TimeoutExpired as error:
                raise MirageTimeout(f'MIRAGE worker timed out after {self.timeout_s:g}s') from error
            except (subprocess.CalledProcessError, OSError) as error:
                details = (root / 'stderr.log').read_bytes()[-8000:].decode(errors='replace')
                raise MirageRuntimeError(f'MIRAGE worker failed: {details or str(error)}') from error
            try:
                values = np.load(root / 'prediction.npy', allow_pickle=False)
                provenance = json.loads((root / 'provenance.json').read_text())
            except (OSError, ValueError) as error:
                raise MirageRuntimeError('MIRAGE worker returned incomplete output artifacts') from error
            return Prediction(values=values, provenance=provenance)
