# Run an outside experiment library in its own environment

Some experiment libraries need packages that do not fit the scoring environment, or must
run at an exact commit with exact package versions. Brain-Score can run such a library in
its own Python environment, in a separate process, while the subject stays in the scoring
process. This page shows how, with a made-up library called Lanternfield.

## When you need this

- The library pins packages that conflict with Brain-Score's packages.
- The library's own code must run unchanged, at one commit, for its scores to count.
- You do not want an install into the environment that scores models.

If the library imports cleanly into the scoring environment, you do not need this: call it
directly from an adapter (`adapter.py`).

## How it works

Two processes. The parent is your Brain-Score process: it holds the subject, the sessions
and the records. The child runs the library with the Python of the library's own
environment and imports no Brain-Score package.

```text
  scoring process (parent)                     library environment (child)
  ------------------------                     ---------------------------
  subject, sessions, records                   worker script + library checkout

  child.ask(command)    --- command  ------->  runs the library
  answer(call)          <-- {"call": ...} ---  the library needs the subject
     run_request(...)   --- reply    ------->  (any number of calls)
  return value          <-- {"done": ...} ---  or {"failed": {...}}
```

Each message is one JSON object on one line, on two pipes that the child finds in
`BRAINSCORE_CHILD_FDS` (`"<read fd>,<write fd>"`), not on standard input and output, so
the library may print. The parent sends a command; the child sends `call` messages, and
the parent answers each with one line (a `reply`); the child ends the command with `done`
or `failed`. When the parent closes its pipe, the child stops. `_child_process.py` holds
the parent's side (`ChildProcess`).

At the first use, `_environment_folder.py` makes one folder for the library:
`checkout/` (the repository at the pinned commit), `env/` (a virtual environment with
the pinned packages only) and `ready.json` (the commit, the requirements' hash, the
Python version and the package list).

## What you write

Three parts, all in `brainscore/harnesses/`. The test
`tests/test_outside_library.py` runs the blocks below as written.

### 1. The description

One `Library` names the library's repository, the full commit, the requirements file and
the files the checkout must have. `python` names the Python that makes the environment;
leave it out to use the scoring process's Python.

```python
# brainscore/harnesses/lanternfield.py
from pathlib import Path

from brainscore.harnesses._environment_folder import Library

LANTERNFIELD = Library(
    name='Lanternfield',
    repository='https://example.org/lanternfield.git',
    commit='4c1d6e0a9b3f27d85e6c0b1a2f4d9e7c3b5a8f60',
    requirements=Path(__file__).with_name('lanternfield_requirements.txt'),
    required_files=('lanternfield/__init__.py',),     # checked after the clone
)
```

The requirements file lists every package of the environment, one `name==version` per
line, with their dependencies, `pip` and `setuptools`. It is installed with `--no-deps`
and checked with `pip check`, so a missing dependency fails at creation.

### 2. The worker script

The child runs `<environment python> <worker script> <checkout>`. The script uses the
standard library only: no Brain-Score package is in the library's environment.

```python
# brainscore/harnesses/_lanternfield_worker.py
import json
import os
import sys
import traceback

read_fd, write_fd = (int(fd) for fd in os.environ['BRAINSCORE_CHILD_FDS'].split(','))
commands, messages = os.fdopen(read_fd, 'rb'), os.fdopen(write_fd, 'wb')


def send(message):
    messages.write((json.dumps(message) + '\n').encode())
    messages.flush()


def ask_subject(question):
    send({'call': {'question': question}})            # the parent asks the subject
    return json.loads(commands.readline())['reply']


sys.path.insert(0, sys.argv[1])                       # the library's checkout
import lanternfield                                   # noqa: E402

while line := commands.readline():                    # empty: the parent closed its pipe
    command = json.loads(line)
    try:
        send({'done': lanternfield.run_trials(command['questions'], ask_subject)})
    except Exception as error:
        send({'failed': {'type': type(error).__name__, 'message': str(error),
                         'traceback': traceback.format_exc()}})
```

### 3. The adapter

The adapter starts the child with the folder's Python and answers each call with one
session of the subject (`run_request` in `adapter.py`). See `adapter.py` for the score
(`adapter_score`) and the recorded run (`run_recorded`).

```python
# brainscore/harnesses/lanternfield.py
import tempfile

import pandas as pd

from brainscore.experiments import CallableProtocol
from brainscore.harnesses._child_process import ChildProcess
from brainscore.harnesses._environment_folder import environment_python, ready_folder
from brainscore.harnesses.adapter import run_request
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Text
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

WORKER = Path(__file__).with_name('_lanternfield_worker.py')


def protocol(questions, folder=None):
    """Lanternfield's trials as a protocol; the folder is made at the first use."""
    folder = ready_folder(LANTERNFIELD, 'Lanternfield.demo', folder)

    def evaluate(subject, context):
        calls = []

        def answer(call):                             # one call: one session of the subject
            calls.append(call)
            stimuli = StimulusSet(pd.DataFrame([{'stimulus_id': f'call-{len(calls)}',
                                                 'text': call['question']}]))
            task = TaskContext(present=('text',), response=Text(max_tokens=8))
            session = run_request(subject, context, f'1.{len(calls)}', stimuli,
                                  ['behavior:text'], task)
            return {'reply': str(session.collect('behavior:text').values.ravel()[0])}

        with tempfile.TemporaryDirectory(prefix='lanternfield-') as logs:
            with ChildProcess([environment_python(folder), WORKER, folder / 'checkout'],
                              log_directory=Path(logs), idle_timeout_s=600) as child:
                return child.ask({'questions': list(questions)}, answer)

    return CallableProtocol('lanternfield:demo', evaluate)
```

## What the library must give

- A git repository and one full commit that holds every file the worker reads.
- A complete, exact list of its packages (every dependency pinned).
- A function the worker can call with a way to reach the subject (here, `ask_subject`),
  so each model call goes to the parent.

## What the machine must give

- Linux or macOS (process groups and pipes); not Windows.
- `git` on `PATH`, and network access to the repository and the package index at the
  first use.
- A Python 3 program for the environment. Brain-Score installs no Python: name one with
  `python=` when the library needs another version, or the scoring Python is used.
- Disk space for the folder. All folders are under `~/.brain-score/environments/<name>/`;
  `BRAINSCORE_ENVIRONMENTS_HOME` moves this root (an absolute path), and `folder=` of
  `ready_folder` names one exact folder.

## What Brain-Score guarantees

- Nothing is installed into the scoring environment. pip runs only inside the folder's
  environment (`--require-virtualenv`), with no pip configuration file and an allowlisted
  set of variables.
- A folder is made completely or not at all: in a temporary folder next to it, checked,
  then published by one rename that never replaces. A folder that exists, or appears
  meanwhile, is never changed.
- One creation at a time (a lock file next to the folder). A killed creation stops its
  git or pip command; the next creation removes its temporary folder. Only a temporary
  folder with Brain-Score's creation record and no `ready.json` is ever removed.
- A changed commit, requirements file or Python version gives a new folder; an old folder
  is left as it is.
- `BS_INSTALL_DEPENDENCIES=no` forbids making a folder: a load with no ready folder stops
  with one error that names the folder and the next step.
- The child has its own process group. It is killed with its group when the parent stops
  waiting (a deadline, an error, the end of the `with` block). No child outlives the block.
- A message longer than 256 MiB, a line that is not JSON, or silence past `idle_timeout_s`
  stops the child with an error that quotes the end of its error output.

## Limits

- Only JSON crosses the pipe: strings, numbers, lists, objects, `true`, `false`, `null`.
  A tuple arrives as a list; bytes and NaN do not cross. Send files as paths.
- An exception crosses as its type name, message and traceback, not as the same class.
- The library's module state lives as long as one child process.
- `ready_folder` does not update a folder. To rebuild one, remove it yourself.

## Test the integration offline

`tests/test_outside_library.py` makes Lanternfield's folder and runs one command through
`ChildProcess`, with no network and no install: stand-ins replace `git` and the Python
that makes the environment and runs pip (`tests/command_stand_ins.py`, `stand_ins`), and
the clone copies a small checkout from the test. Copy it for a new library: give the
stand-ins your checkout and commit, and run your worker on a stand-in of the library.
