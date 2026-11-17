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
        to_parent_read, to_parent_write = os.pipe()
        to_child_read, to_child_write = os.pipe()
        self.env['BRAINSCORE_CHILD_FDS'] = f'{to_child_read},{to_parent_write}'
        try:
            with (self.log_directory / 'stdout.log').open('wb') as stdout, \
                    (self.log_directory / 'stderr.log').open('wb') as stderr:
                self.process = subprocess.Popen(
                    self.command, env=self.env, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                    pass_fds=(to_child_read, to_parent_write), start_new_session=True)
        except BaseException:
            for fd in (to_parent_read, to_parent_write, to_child_read, to_child_write):
                os.close(fd)
            raise
        os.close(to_child_read)
        os.close(to_parent_write)
        self._read, self._write = to_parent_read, to_child_write
        return self

    def __exit__(self, kind, error, traceback) -> None:
        self._close_pipes()
        if self.process is None:
            return
        try:
            self.process.wait(timeout=0 if error is not None else STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        finally:
            self.kill()

    def kill(self) -> None:
        """Kill the child's process group (if it still runs) and reap the child."""
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process.wait()

    def wait(self, timeout_s: float) -> int:
        """Wait for the child to exit; ``subprocess.TimeoutExpired`` after ``timeout_s``
        (the child is killed), ``CalledProcessError`` on a nonzero exit."""
        try:
            returncode = self.process.wait(timeout=timeout_s)
        except BaseException:
            self.kill()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, self.command)
        return returncode

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
        """The end of the child's error output."""
        try:
            return (self.log_directory / 'stderr.log').read_bytes()[-limit:].decode(errors='replace')
        except OSError:
            return ''

    def _send(self, message: Any) -> None:
        data = (json.dumps(message, allow_nan=False) + '\n').encode()
        try:
            while data:
                data = data[os.write(self._write, data):]
        except BrokenPipeError:
            raise self._fail('the child closed its pipe') from None

    def _receive(self) -> dict:
        deadline = None if self.idle_timeout_s is None else time.monotonic() + self.idle_timeout_s
        while b'\n' not in self._buffer:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self._read], [], [], remaining)
            if not ready:
                raise self._fail(f'the child said nothing for {self.idle_timeout_s:g} seconds')
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
