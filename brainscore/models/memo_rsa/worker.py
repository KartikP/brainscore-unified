"""Small JSON-lines client for a separately installed memo/JAX runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import sys


class MemoWorkerError(RuntimeError):
    """The memo worker failed or returned an invalid response."""


class MemoWorkerTimeout(MemoWorkerError):
    """The memo worker did not answer within its configured timeout."""


class MemoWorker:
    """Persistent worker client; each call sends one batch of utterances."""

    def __init__(self, python_executable: str, *, timeout_s: float = 30.0,
                 worker_script: str | Path | None = None):
        if not isinstance(python_executable, str) or not python_executable:
            raise ValueError('python_executable must name a Python interpreter')
        if timeout_s <= 0:
            raise ValueError('timeout_s must be positive')
        self.python_executable = python_executable
        self.timeout_s = float(timeout_s)
        self.worker_script = Path(worker_script or Path(__file__).with_name('memo_worker.py'))
        self._process = None
        self.runtime = None

    def _start(self):
        environment = os.environ.copy()
        environment['JAX_PLATFORMS'] = 'cpu'
        try:
            self._process = subprocess.Popen(
                [self.python_executable, '-u', str(self.worker_script)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=0, env=environment,
            )
        except OSError as error:
            raise MemoWorkerError(f'could not start memo worker: {error}') from error
        hello = self._request({'op': 'hello'})
        if tuple(hello.get('python_major_minor', (0, 0))) < (3, 12):
            self.close()
            raise MemoWorkerError('memo worker requires Python >=3.12')
        if not hello.get('memo_version') or not hello.get('jax_version'):
            self.close()
            raise MemoWorkerError('memo worker did not report memo and JAX versions')
        self.runtime = hello

    def _request(self, request):
        process = self._process
        if process is None or process.poll() is not None:
            raise MemoWorkerError('memo worker is not running')
        try:
            process.stdin.write((json.dumps(request, allow_nan=False) + '\n').encode())
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.close()
            raise MemoWorkerError('memo worker closed its input') from error
        ready, _, _ = select.select([process.stdout], [], [], self.timeout_s)
        if not ready:
            self.close()
            raise MemoWorkerTimeout(f'memo worker timed out after {self.timeout_s:g}s')
        line = process.stdout.readline()
        if not line:
            code = process.poll()
            self.close()
            raise MemoWorkerError(f'memo worker exited without a response (exit code {code})')
        try:
            response = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self.close()
            raise MemoWorkerError('memo worker returned invalid JSON') from error
        if not isinstance(response, dict) or response.get('ok') is not True:
            message = (response.get('error', 'invalid response') if isinstance(response, dict)
                       else 'invalid response')
            self.close()
            raise MemoWorkerError(f'memo worker error: {message}')
        return response

    def predict(self, *, depth: int, beta: float,
                utterances: list[str]) -> list[list[float]]:
        if self._process is None:
            self._start()
        response = self._request({
            'op': 'predict', 'depth': depth, 'beta': beta,
            'utterances': utterances,
        })
        if not isinstance(response.get('runtime'), dict) or not isinstance(
                response.get('probabilities'), list):
            self.close()
            raise MemoWorkerError('memo worker returned an incomplete prediction response')
        self.runtime = response['runtime']
        return response['probabilities']

    def close(self) -> None:
        process, self._process = self._process, None
        self.runtime = None
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    process.stdin.write(b'{"op":"shutdown"}\n')
                    process.stdin.flush()
                    process.wait(timeout=min(self.timeout_s, 2.0))
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait(timeout=2.0)
        finally:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()


__all__ = ['MemoWorker', 'MemoWorkerError', 'MemoWorkerTimeout']
