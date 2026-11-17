"""Run one command, and stop it when the process that started it dies.

``<python> -I _supervise.py <command> [arguments...]``, started by ``ChildProcess``: the
command runs in this process's group; this process exits with the command's exit status. When
the starting process's end of the pipe closes (it was killed, or it stopped waiting), the
whole group is killed, this process with it. Standard library only.
"""
import os
import select
import signal
import subprocess
import sys

if __name__ == '__main__':
    parent = int(os.environ['BRAINSCORE_CHILD_FDS'].split(',')[0])
    command = subprocess.Popen(sys.argv[1:])
    while True:
        ready, _, _ = select.select([parent], [], [], 0.2)
        if ready and not os.read(parent, 1):
            os.killpg(os.getpgrp(), signal.SIGKILL)
        status = command.poll()
        if status is not None:
            sys.exit(status if status >= 0 else 128 - status)
