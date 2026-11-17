"""A child process with its own Python: started, talked to and always reaped.

The one process control of the harnesses and runtimes that run outside code in another
environment (CogGym's evaluator, MIRAGE's upstream pipeline). The child gets its own
process group, writes its output to two log files, and is killed with its group when the
parent stops waiting (a deadline, an error, an interruption); no child outlives its
``with`` block.

A conversation uses two dedicated pipes, not stdin and stdout (outside code may print).
The child finds them in ``BRAINSCORE_CHILD_FDS`` as ``"<read fd>,<write fd>"``. Every
message is one JSON object on one line:

- the parent sends a command (``ask``);
- the child may send ``{"call": ...}`` any number of times; the parent answers each with
  the line its ``answer`` function returns;
- the child ends the command with ``{"done": <value>}`` or ``{"failed": {"type",
  "message", "traceback"}}``.

The parent closes its end to tell the child to stop.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
from typing import Any, Callable, Mapping, Optional, Sequence

# Seconds a child has to exit after its pipe is closed, before it is killed.
STOP_GRACE_SECONDS = 5.0
# The longest line either side may send: room for a call with many pictures as base64 text.
MAX_LINE_BYTES = 256 * 1024 * 1024


class ChildFailed(RuntimeError):
    """The child failed a command: its error type, message and traceback are in the text."""

    def __init__(self, message: str, error_type: str = '', child_traceback: str = '') -> None:
        super().__init__(message)
        self.error_type, self.child_traceback = error_type, child_traceback


class ChildProcess:
    """``with ChildProcess(command, log_directory=...) as child: child.ask(...)`` or ``child.wait(...)``."""

    def __init__(self, command: Sequence[str], *, log_directory: Path,
                 env: Optional[Mapping[str, str]] = None, idle_timeout_s: Optional[float] = None) -> None:
        self.command = [str(part) for part in command]
        self.log_directory = Path(log_directory)
        self.env = dict(os.environ if env is None else env)
        self.idle_timeout_s = idle_timeout_s
        self.process: Optional[subprocess.Popen] = None
        self._read = self._write = None
        self._buffer = b''

    def __enter__(self) -> 'ChildProcess':
        self.log_directory.mkdir(parents=True, exist_ok=True)
        acquired = []           # every descriptor taken so far: closed if a later step fails
        try:
            acquired += os.pipe()
            acquired += os.pipe()
            to_parent_read, to_parent_write, to_child_read, to_child_write = acquired
            self.env['BRAINSCORE_CHILD_FDS'] = f'{to_child_read},{to_parent_write}'
            with (self.log_directory / 'stdout.log').open('wb') as stdout, \
                    (self.log_directory / 'stderr.log').open('wb') as stderr:
                self.process = subprocess.Popen(
                    self.command, env=self.env, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                    pass_fds=(to_child_read, to_parent_write), start_new_session=True)
        except BaseException:
            for fd in acquired:
                os.close(fd)
            raise
        os.close(to_child_read)
        os.close(to_parent_write)
        self._read, self._write = to_parent_read, to_child_write
        os.set_blocking(self._write, False)       # a send waits under the deadline, not forever
        return self

    def __exit__(self, kind, error, traceback) -> None:
        self._close_pipes()
        if self.process is None:
            return
        try:
            # The child stops when its pipe closes; give it the grace time, unless this is an error.
            self._leader_exited(0 if error is not None else STOP_GRACE_SECONDS)
        finally:
            self.kill()

    def kill(self) -> None:
        """Kill the child's whole process group, also when the child itself has exited, then
        reap the child.

        Where the platform can wait without reaping (``os.waitid``, Linux), the group is killed
        while the exited child still holds its id, so the id cannot belong to another group. On
        macOS (Python 3.11 has no ``os.waitid``) a child that exited on its own is reaped first
        and its group is killed at once after. A descendant that left the group (``setsid``) is
        not killed.
        """
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self.process.wait()

    def _leader_exited(self, timeout_s: float) -> bool:
        """Whether the child exited within ``timeout_s``; it is not reaped where the platform allows."""
        if not hasattr(os, 'waitid'):
            try:
                self.process.wait(timeout=timeout_s)
                return True
            except subprocess.TimeoutExpired:
                return False
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                    return True
            except ChildProcessError:
                return True                 # already reaped
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)

    def wait(self, timeout_s: float) -> int:
        """Wait for the child to exit; ``subprocess.TimeoutExpired`` after ``timeout_s``
        (the child is killed), ``CalledProcessError`` on a nonzero exit. The child's process
        group is killed in both cases."""
        try:
            exited = self._leader_exited(timeout_s)
        except BaseException:
            self.kill()
            raise
        self.kill()
        if not exited:
            raise subprocess.TimeoutExpired(self.command, timeout_s)
        if self.process.returncode:
            raise subprocess.CalledProcessError(self.process.returncode, self.command)
        return self.process.returncode

    def ask(self, command: Any, answer: Callable[[Any], Any] = None) -> Any:
        """Send ``command``; answer each call of the child; return its ``done`` value.

        A child that exits, writes a line that is not JSON, or says nothing for
        ``idle_timeout_s`` seconds while the parent waits for it is killed, and
        ``ChildFailed`` names what happened and quotes its log.
        """
        self._send(command)
        while True:
            message = self._receive()
            if 'call' in message:
                if answer is None:
                    raise self._fail('the child made a call, and this command takes none')
                self._send(answer(message['call']))
            elif 'done' in message:
                return message['done']
            elif 'failed' in message:
                failed = message['failed']
                raise ChildFailed(f"{failed.get('type')}: {failed.get('message')}",
                                  failed.get('type', ''), failed.get('traceback', ''))
            else:
                raise self._fail(f'the child sent a message of no known kind: {sorted(message)}')

    def log_tail(self, limit: int = 4000) -> str:
        """The last ``limit`` bytes of the child's error output (the rest is not read)."""
        try:
            with (self.log_directory / 'stderr.log').open('rb') as log:
                log.seek(max(0, log.seek(0, os.SEEK_END) - limit))
                return log.read().decode(errors='replace')
        except OSError:
            return ''

    def _send(self, message: Any) -> None:
        """Write one line; a child that does not read it within ``idle_timeout_s`` is killed."""
        data = (json.dumps(message, allow_nan=False) + '\n').encode()
        if len(data) > MAX_LINE_BYTES:
            raise ValueError(f'a message of {len(data)} bytes is longer than the limit of {MAX_LINE_BYTES}')
        deadline = None if self.idle_timeout_s is None else time.monotonic() + self.idle_timeout_s
        try:
            while data:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                _, ready, _ = select.select([], [self._write], [], remaining)
                if not ready:
                    raise self._fail(f'the child read nothing for {self.idle_timeout_s:g} seconds')
                try:
                    data = data[os.write(self._write, data):]
                except BlockingIOError:
                    continue
        except BrokenPipeError:
            raise self._fail('the child closed its pipe') from None

    def _receive(self) -> dict:
        deadline = None if self.idle_timeout_s is None else time.monotonic() + self.idle_timeout_s
        while b'\n' not in self._buffer:
            if len(self._buffer) > MAX_LINE_BYTES:
                raise self._fail(f'the child wrote a line longer than {MAX_LINE_BYTES} bytes')
            remaining = None if deadline is None else deadline - time.monotonic()
            # The deadline holds for the whole line, also when the child writes it slowly.
            ready = remaining is None or remaining > 0
            if ready:
                ready, _, _ = select.select([self._read], [], [], remaining)
            if not ready:
                raise self._fail(f'the child said nothing complete for {self.idle_timeout_s:g} seconds')
            chunk = os.read(self._read, 1 << 16)
            if not chunk:
                raise self._fail('the child exited')
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b'\n', 1)
        try:
            message = json.loads(line)
        except ValueError:
            raise self._fail('the child wrote a line that is not JSON') from None
        if not isinstance(message, dict):
            raise self._fail('the child wrote a message that is not an object')
        return message

    def _fail(self, what: str) -> ChildFailed:
        self.kill()
        tail = self.log_tail()
        return ChildFailed(f'{what} (command {self.command[:2]}); its error output ends with:\n{tail}'
                           if tail else f'{what} (command {self.command[:2]})')

    def _close_pipes(self) -> None:
        for name in ('_write', '_read'):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)
