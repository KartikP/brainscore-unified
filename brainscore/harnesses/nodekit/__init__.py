"""Any NodeKit (nkl) program as a closed-loop environment: a subject takes part as a person does, with the
keyboard and the pointer, and sees only the screen.

A program is an nkl bundle folder (its ``program.json``: the lowered program, its arguments and the
files of its media). nkl runs it in a child process with its own environment (``_environment_folder``,
Python 3.13): the checkout is NodeKit's task repository at a pinned commit, which carries the nkl wheel
and the bundles. nkl drives its own headless Chrome for Testing (downloaded by nkl at the first run).
Nothing of nkl is imported here.

Each step: the row holds the frames since the last step on ``vision:video`` ((T, 384, 512, 3) uint8: one at
each frame time the clock passed, 60 frames a second (``FRAME_MS``; nkl states no display rate), at half
the screen size), the screen at the end of the step on ``vision:image`` ((H, W, 3) uint8, 1024 x 768 by
default) and the acts on ``instruction``; the subject answers on the act form the benchmark asks for
(``form``, Contract Rules 5.15), ``motor:key`` or ``motor:pointer``, with a list of acts, each a mapping
``{'act': <name>, <fields>}`` (core's ``MOTOR_ACTS``, which also checks them): key acts ``press``,
``release`` (a W3C ``KeyboardEvent.code``, such as ``Space`` or ``ArrowLeft``) and ``type``; pointer acts
``move``, ``down``, ``up`` and ``stroke``; ``wait`` on both. Positions are pixels of the last picture: the
origin at the top left, x to the right, y down.
``nkl_acts`` is the one place that turns them into nkl's acts (scene units: origin at the centre, y up).
The program's clock moves only by the acts (``wait`` passes time; a key or button act takes 1 ms, a stroke
as nkl's motor allows): the subject's own compute time is not counted. Each step runs to at least the next
frame time (the worker adds the wait), so a step with no act (an empty list, valid) or with an answer that
could not be read (None, counted in ``invalid_steps``) also moves the clock by one frame. The run ends
when the program ends (``terminal``; ``result`` is then the program's result as value JSON); when its
simulated time passes ``max_ms`` or its steps pass ``max_steps``, the harness leaves the run (nkl's Leave)
and it ends ``truncated`` with no result; an nkl error stops the session with the error. Times in the
result (response times) are simulated times.

Each run's nkl trace is written to ``<record>/<folder>/submissions/<name>.json``, the form of a
participant's submission in the task repository, so ``python -m brainscore.harnesses.nodekit replay
<record>`` opens the repository's replay viewer on it (``replay``), as on the participants' runs.

Limit: a subject that takes ``vision:image`` and not ``vision:video`` sees only the screen at the end of
each step: a stimulus shorter than its steps (a brief patch) can pass unseen.

Limit: a session takes its acts on one channel: a program that needs key and pointer acts in one run (type
a text, then click) runs with the one form its benchmark asks for, and a step that needs the other form
cannot be taken.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from brainscore_core.io_catalog import MOTOR_ACTS
from brainscore.harnesses._child_process import ChildProcess
from brainscore.harnesses._environment_folder import Library, _RUNTIME_VARIABLES, environment_python, ready_folder

# NodeKit's task repository: the nkl wheel (vendor/), the bundles (bundles/) and the participants' trials
# (prolific/processed/). Private: the user needs read access. Nothing of it is copied into Brain-Score.
REPOSITORY = 'https://github.com/intelligence-observatory/umi-nkl.git'
COMMIT = '46ff367543ed5a209147ad9b8e6275ce81059d38'
NKL_WHEEL = 'vendor/nkl-0.1.0-py3-none-any.whl'
SCREEN = (1024, 768)
FRAME_MS = 1000 / 60            # the frame times of vision:video: 60 frames a second
VIDEO = (512, 384)              # the size of a vision:video frame
# The participants' folder of each bundle in the task repository: the replay viewer's names.
VIEWER_FOLDERS = {'drifting-gabor': 'gabor', 'triplet-similarity': 'triplets', 'face-arrangement': 'faces'}
FORMS = ('motor:key', 'motor:pointer')
# The act of nkl.sim.actions for each act name.
_NKL_ACTS = {'press': 'KeyPress', 'release': 'KeyRelease', 'type': 'TypeText', 'move': 'PointerMove',
             'down': 'PointerDown', 'up': 'PointerUp', 'stroke': 'PointerStroke', 'wait': 'Wait'}
_WORKER = Path(__file__).with_name('_worker.py')
# Seconds the child may take for one command (a first reset starts nkl's server and Chrome).
_CHILD_SILENCE_S = 600.0


def source() -> Path:
    """The task repository at ``COMMIT`` in ``<data home>/sources/nodekit/umi-nkl``, cloned at the first use."""
    from brainscore.data.local import data_home
    clone = data_home() / 'sources' / 'nodekit' / 'umi-nkl'

    def git(*arguments):
        return subprocess.run(['git', '-C', str(clone), *arguments], capture_output=True, text=True)
    try:
        if not (clone / '.git').exists():
            clone.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--quiet', '--no-checkout', REPOSITORY, str(clone)],
                           check=True, capture_output=True, text=True)
        if git('rev-parse', 'HEAD').stdout.strip() != COMMIT or not (clone / NKL_WHEEL).is_file():
            if git('cat-file', '-e', f'{COMMIT}^{{commit}}').returncode != 0:
                subprocess.run(['git', '-C', str(clone), 'fetch', '--quiet', 'origin', COMMIT], check=True,
                               capture_output=True, text=True)
            subprocess.run(['git', '-C', str(clone), '-c', 'core.hooksPath=/dev/null', 'checkout', '--quiet',
                            COMMIT], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f'Brain-Score could not clone NodeKit\'s task repository {REPOSITORY} at {COMMIT} into {clone}: '
            f'{(error.stderr or "").strip()[-500:]}. The repository is private. Next step: ask its owners for '
            f'read access and set up git credentials for github.com, then run again.') from error
    return clone


def library() -> Library:
    """nkl's environment: the clone of the task repository (cloned again from the local clone) and its wheel."""
    return Library(name='nodekit', repository=str(source()), commit=COMMIT,
                   requirements=Path(__file__).with_name('requirements.txt'), required_files=(NKL_WHEEL,),
                   python='python3.13', checkout_requirements=True)


def bundle(name: str) -> Path:
    """The bundle ``bundles/<name>`` of the task repository (for example ``drifting-gabor``)."""
    folder = source() / 'bundles' / name
    if not (folder / 'program.json').is_file():
        names = sorted(path.parent.name for path in (source() / 'bundles').glob('*/program.json'))
        raise ValueError(f'{name!r} is not a bundle of the task repository; the bundles are {names}.')
    return folder


def nkl_acts(acts, screen) -> list:
    """The acts as nkl's (``nkl.sim.actions`` names and fields): pixels of a ``screen`` (width, height)
    picture, origin top left and y down, become scene units, origin at the centre and y up (device ratio 1)."""
    width, height = screen

    def scene(x, y):
        return x - width / 2, height / 2 - y
    converted = []
    for act in acts:
        fields = {key: value for key, value in act.items() if key != 'act'}
        if 'x' in fields:
            fields['x'], fields['y'] = scene(fields['x'], fields['y'])
        if 'points' in fields:
            fields['points'] = [list(scene(*point)) for point in fields['points']]
        converted.append({'act': _NKL_ACTS[act['act']], **fields})
    return converted


class NodeKitEnvironment:
    """One run of the nkl program in the bundle folder ``program``, with ``seed`` (a bundle's trials are
    already dealt; the seed is nkl's run seed), on a screen of ``screen`` pixels; the subject acts on
    ``form`` (``motor:key`` or ``motor:pointer``)."""

    def __init__(self, program, *, form: str, seed: int = 0, screen=SCREEN, max_ms: float = 3_600_000, max_steps: int = 20000,
                 record: Optional[Path] = None, name: str = 'run', folder: Optional[Path] = None,
                 log_directory: Optional[Path] = None):
        self.program, self.seed, self.screen = Path(program), seed, tuple(screen)
        self.max_ms, self.max_steps, self.record, self.name = max_ms, max_steps, record, name
        if not (self.program / 'program.json').is_file():
            raise ValueError(f'{self.program} is not an nkl bundle: it holds no program.json.')
        if form not in FORMS:
            raise ValueError(f'A NodeKit subject acts on {sorted(FORMS)}, not {form!r} (Contract Rules 5.15).')
        self.form = form
        self.folder = ready_folder(library(), f'NodeKit {self.program.name}', folder)
        self.log_directory = Path(log_directory) if log_directory else self.folder / 'logs'
        self._child, self.steps, self.t_ms, self.result, self.invalid_steps, self.trace = None, 0, 0.0, None, 0, None

    def instruction(self) -> str:
        acts = '; '.join(f'{name}({", ".join(sorted(required) + [f"{key} (optional)" for key in sorted(optional)])})'
                         for name, (required, optional) in MOTOR_ACTS[self.form.split(':')[1]].items())
        return (f'You take part in a task on a screen of {self.screen[0]} x {self.screen[1]} pixels; you see the '
                f'screen as a picture. Act on {self.form} with a list of acts, each {{"act": name, fields}}: {acts}. '
                f'Positions are pixels of the picture: the origin at the top left, x to the right, y down. Time '
                f'passes only by your acts: wait(ms) lets time pass. Each step shows the frames since the last '
                f'step and the screen now.')

    def _row(self, answer: dict):
        self.t_ms = answer['t_ms']
        if answer['ended']:
            return {}

        def decoded(part):
            return np.frombuffer(base64.b64decode(part['data']), np.uint8).reshape(part['shape'])
        row = {'vision:image': decoded(answer['picture'])[0], 'instruction': self.instruction()}
        if answer['frames']:
            row['vision:video'] = decoded(answer['frames'])
        return row

    def _ended(self, row, reason):
        if reason == 'truncated':
            self._child.ask({'leave': True})
        self.result = self._child.ask({'result': True})
        if self.record is not None:
            folder = Path(self.record) / VIEWER_FOLDERS.get(self.program.name, self.program.name) / 'submissions'
            folder.mkdir(parents=True, exist_ok=True)
            self.trace = Path(self._child.ask({'trace': str(folder / f'{self.name}.json')}))
        return row, reason

    def reset(self, **_):
        self.close()
        home = self.folder.parent / 'nkl-home'      # nkl's data and its Chrome download
        environment = {name: os.environ[name] for name in _RUNTIME_VARIABLES if name in os.environ}
        environment.update(PYTHONNOUSERSITE='1', HOME=str(home), NKL_HOME=str(home / 'nkl'))
        self._child = ChildProcess([environment_python(self.folder), _WORKER, self.program, self.seed, *self.screen,
                                    FRAME_MS, *VIDEO],
                                   env=environment, log_directory=self.log_directory,
                                   idle_timeout_s=_CHILD_SILENCE_S).__enter__()
        self.steps, self.result, self.invalid_steps, self.trace = 0, None, 0, None
        answer = self._child.ask({'reset': True})
        row = self._row(answer)
        return self._ended(row, 'terminal') if answer['ended'] else row

    def step(self, outputs):
        acts = outputs[self.form]          # checked by the session (core's MOTOR_ACTS)
        if acts is None:
            self.invalid_steps += 1
        answer = self._child.ask({'step': nkl_acts(acts or [], self.screen)})
        self.steps += 1
        row = self._row(answer)
        if answer['ended']:
            return self._ended(row, 'terminal')
        if self.t_ms >= self.max_ms or self.steps >= self.max_steps:
            return self._ended(row, 'truncated')
        return row, None

    def clock(self):
        return {'t_ms': self.t_ms, 'time_source': 'nkl simulation clock', 'time_inferred': False}

    def close(self):
        if self._child is not None:
            self._child.__exit__(None, None, None)
            self._child = None


def play(subject, program, *, form: str, **settings):
    """One run of ``program`` (a bundle folder) by ``subject`` on the act form ``form`` through
    ``environment_session``:
    ``(result, environment, session)``; ``result`` is None when the run did not complete. With ``record=``
    the run's trace is kept there under the subject's identifier and the time."""
    import time
    from brainscore_core.streaming_helpers import environment_session
    settings.setdefault('name', f"{getattr(subject, 'identifier', 'subject')}-{time.strftime('%Y%m%dT%H%M%S')}")
    environment = NodeKitEnvironment(program, form=form, **settings)
    try:
        session = environment_session(environment, [form], action_channel=form)
        subject.participate(session)
        return environment.result, environment, session
    finally:
        environment.close()


def replay(record, port: int = 5310) -> None:
    """Open the task repository's replay viewer on ``record`` (a folder of ``<folder>/submissions/*.json``
    traces, a subject's or the participants' ``prolific/``), served on 127.0.0.1 with the pinned bundles."""
    folder = ready_folder(library(), 'NodeKit replay')
    checkout = folder / 'checkout'
    subprocess.run([str(environment_python(folder)), str(checkout / 'src' / 'umi_nkl' / 'viewer.py'),
                    '--prolific', str(Path(record).resolve()), '--bundles', str(checkout / 'bundles'),
                    '--port', str(port)], check=True)
