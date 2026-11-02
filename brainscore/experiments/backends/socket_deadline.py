"""Deadlines for blocking websocket writes, exchanges and closes.

Standard library only and Python 3.8 syntax: examples/libero/evaluate.py loads
this file by path in the LIBERO environment, without the brainscore package.

websockets' synchronous send() has no timeout, and close() can wait behind a
blocked send or a receiver paused by a full message queue. When time runs out
a deadline calls the pinned library's (15.0.1) own termination step,
close_socket(), from a timer thread. It shuts the socket down before taking
the protocol lock, so a send blocked while holding that lock fails and
releases it; it then closes the socket and the message queue, which resumes a
paused receiver and guarantees the receiver thread ends. The connection is
unusable afterwards and the caller must not retry. Nothing on the peer is
cancelled: a request already delivered may still be processed there.
"""
from contextlib import contextmanager
import logging
import math
import threading
import time
from typing import Any, Callable, Iterator, Optional

# Conservative wait for a closing handshake; matches websockets' own default.
CLOSE_TIMEOUT = 10.0


def positive_seconds(seconds: float) -> float:
    """Return seconds if it is a positive, finite int or float; else raise ValueError."""
    if not (type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0):
        raise ValueError('Timeouts must be positive, finite numbers of seconds')
    return seconds


@contextmanager
def deadline(connection: Any, seconds: float) -> Iterator[Callable[[], float]]:
    """Bound the block; yields remaining(), the seconds left for a recv timeout.

    If the time runs out first, the connection is terminated and TimeoutError
    is raised, chained to whatever the interrupted call raised. A block that
    finishes as the deadline fires counts as timed out: the connection may
    already be terminated. The timer thread is cancelled and joined before
    returning, so no deadline outlives its block.
    """
    positive_seconds(seconds)
    end = time.monotonic() + seconds
    lock = threading.Lock()
    state = {'done': False, 'fired': False}

    def abort():
        with lock:
            if state['done']:
                return
            state['fired'] = True
        connection.close_socket()

    def finish():
        with lock:
            state['done'] = True
        timer.cancel()
        timer.join()
        return state['fired']

    timer = threading.Timer(seconds, abort)
    timer.daemon = True
    timer.start()
    try:
        yield lambda: max(end - time.monotonic(), 0.0)
    except BaseException as error:
        if finish():
            raise TimeoutError('Connection deadline of %g s exceeded' % seconds) from error
        raise
    if finish():
        raise TimeoutError('Connection deadline of %g s exceeded' % seconds)


def close(connection: Any, seconds: Optional[float] = None, **kwargs: Any) -> bool:
    """Close within seconds (default CLOSE_TIMEOUT); True if clean. Never raises an Exception.

    Errors are logged rather than raised so cleanup never replaces the error
    that caused it. After this returns the connection is closed or terminated.
    """
    try:
        with deadline(connection, CLOSE_TIMEOUT if seconds is None else seconds):
            connection.close(**kwargs)
        return True
    except Exception as error:
        logging.getLogger(__name__).warning('Connection did not close cleanly: %r', error)
        try:
            connection.close_socket()
        except Exception:
            pass
        return False
