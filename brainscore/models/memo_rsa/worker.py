"""Small JSON-lines client for a separately installed memo/JAX runtime."""

from __future__ import annotations

import json
import os
import math
from pathlib import Path
import select
import subprocess
import time


class MemoWorkerError(RuntimeError):
    """The memo worker failed or returned an invalid response."""


class MemoWorkerTimeout(MemoWorkerError):
    """The memo worker did not answer within its configured timeout."""


class MemoWorker:
    """Persistent worker client; each call sends one batch of utterances."""

    def __init__(self, python_executable: str, *, timeout_s: float = 30.0,
                 worker_script: str | Path | None = None) -> None:
        if not isinstance(python_executable, str) or not python_executable:
            raise ValueError('python_executable must name a Python interpreter')
        if not math.isfinite(timeout_s) or timeout_s <= 0:
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
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)
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
            self.close(force=True)
            raise MemoWorkerError('memo worker is not running')
        pending = memoryview((json.dumps(request, allow_nan=False) + '\n').encode())
        deadline = time.monotonic() + self.timeout_s
        received = bytearray()
        try:
            while pending or b'\n' not in received:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MemoWorkerTimeout(f'memo worker timed out after {self.timeout_s:g}s')
                readable, writable, _ = select.select(
                    [] if pending else [process.stdout],
                    [process.stdin] if pending else [],
                    [],
                    remaining,
                )
                if writable:
                    try:
                        count = os.write(process.stdin.fileno(), pending[:65536])
                    except BlockingIOError:
                        continue
                    pending = pending[count:]
                if readable:
                    try:
                        chunk = os.read(process.stdout.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        raise MemoWorkerError('memo worker exited without a complete response')
                    received.extend(chunk)
        except (MemoWorkerError, OSError, ValueError) as error:
            self.close(force=True)
            if isinstance(error, MemoWorkerError):
                raise
            raise MemoWorkerError(f'memo worker transport failed: {error}') from error
        line = bytes(received)
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

    def close(self, *, force: bool = False) -> None:
        process, self._process = self._process, None
        self.runtime = None
        if process is None:
            return
        try:
            if process.poll() is None:
                if force:
                    process.kill()
                    process.wait(timeout=2.0)
                    return
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
