"""Any NodeKit (nkl) program as a closed-loop environment: a subject takes part as a person does, with the
keyboard and the pointer, and sees only the screen.

A program is an nkl bundle folder (its ``program.json``: the lowered program, its arguments and the
files of its media). nkl runs it in a child process with its own environment (``_environment_folder``,
Python 3.13): the checkout is NodeKit's task repository at a pinned commit, which carries the nkl wheel
and the bundles. nkl drives its own headless Chrome for Testing (downloaded by nkl at the first run).
Nothing of nkl is imported here.

Each step: the row holds the screen picture on ``vision:image`` ((H, W, 3) uint8, 1024 x 768 by default)
and the act forms on ``instruction``; the subject answers on ``motor`` with a list of acts, each a mapping
``{'act': <name>, <fields>}`` with nkl's own act names (``ACTS``). The program's clock moves only by the
acts (a ``Wait`` passes time; a key or button act takes 1 ms): the clock stands still while the subject
thinks. Positions are scene units: the origin at the screen's centre, x to the right, y up; one unit is
one pixel of the picture. The run ends when the program ends (``terminal``) or after ``max_steps``
(``truncated``); ``result`` is then the program's result as value JSON.

``program_needs`` is the one place that reads what a program needs (key acts, pointer acts) from the
program itself; the act form of the environment follows from it, and a subject that sends an act the
program does not read fails at its first action with the forms the program takes.

Limit: the act forms are not declared by a channel: a subject that answers on ``motor`` with an act the
program does not take fails at its first action (an error, not N/A).
"""
from __future__ import annotations

import base64
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from brainscore.harnesses._child_process import ChildProcess
from brainscore.harnesses._environment_folder import Library, _RUNTIME_VARIABLES, environment_python, ready_folder

# NodeKit's task repository: the nkl wheel (vendor/), the bundles (bundles/) and the participants' trials
# (prolific/processed/). Private: the user needs read access. Nothing of it is copied into Brain-Score.
REPOSITORY = 'https://github.com/intelligence-observatory/umi-nkl.git'
COMMIT = '46ff367543ed5a209147ad9b8e6275ce81059d38'
NKL_WHEEL = 'vendor/nkl-0.1.0-py3-none-any.whl'
SCREEN = (1024, 768)
# nkl's act names (nkl.sim.actions) by the input they need, with their fields: name -> (required, optional).
ACTS = {
    'key': {'KeyPress': ({'key'}, set()), 'KeyRelease': ({'key'}, set()), 'TypeText': ({'text'}, set())},
    'pointer': {'PointerMove': ({'x', 'y'}, set()), 'PointerDown': (set(), set()), 'PointerUp': (set(), set()),
                'PointerStroke': ({'points'}, {'force'})},
    'time': {'Wait': ({'ms'}, set())},
}
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


def program_needs(program: Path) -> frozenset:
    """The inputs a program reads, from its own imports: ``'key'`` (keyboard), ``'pointer'``; time always."""
    imports = set(json.loads((Path(program) / 'program.json').read_text())['program']['imports'])
    reads = {'key': {'keyboard.inputs', 'keyboard.state'}, 'pointer': {'pointer.inputs', 'pointer.state'}}
    return frozenset({'time'} | {need for need, names in reads.items() if imports & names})


class NodeKitActs:
    """The ``motor`` payload of a program: a list of acts (or one act), each ``{'act': name, fields}``
    with a name of ``ACTS`` for an input the program reads; None is no act (an answer that could not be
    read). The key names and the values are checked by nkl."""

    def __init__(self, needs: frozenset, task: str):
        self.forms = {name: fields for need in sorted(needs) for name, fields in ACTS[need].items()}
        self.task = task

    def validate(self, value):
        if value is None:
            return None
        acts = [value] if isinstance(value, dict) else value
        if not isinstance(acts, (list, tuple)):
            raise ValueError(self._expected(f'a {type(value).__name__}'))
        checked = []
        for act in acts:
            name = act.get('act') if isinstance(act, dict) else None
            if name not in self.forms:
                raise ValueError(self._expected(repr(act)))
            required, optional = self.forms[name]
            fields = set(act) - {'act'}
            if not required <= fields <= required | optional:
                raise ValueError(self._expected(f'{name} with fields {sorted(fields)}'))
            for key in fields & {'x', 'y', 'ms', 'force'}:
                if isinstance(act[key], bool) or not isinstance(act[key], (int, float)) or not math.isfinite(act[key]):
                    raise ValueError(self._expected(f'{name} with {key}={act[key]!r}'))
            checked.append(dict(act))
        return checked

    def _expected(self, got: str) -> str:
        return (f'{self.task} takes acts {self.describe()["acts"]} (a list of '
                f'{{"act": name, fields}}); got {got}.')

    def describe(self):
        return {'kind': 'nodekit acts', 'task': self.task,
                'acts': {name: sorted(required) + [f'{key} (optional)' for key in sorted(optional)]
                         for name, (required, optional) in self.forms.items()},
                'positions': 'scene units: origin at the screen centre, x right, y up, one unit one pixel'}


class NodeKitEnvironment:
    """One run of the nkl program in the bundle folder ``program``, with ``seed`` (a bundle's trials are
    already dealt; the seed is nkl's run seed), on a screen of ``screen`` pixels."""

    def __init__(self, program, *, seed: int = 0, screen=SCREEN, max_steps: int = 3000,
                 folder: Optional[Path] = None, log_directory: Optional[Path] = None):
        self.program, self.seed, self.screen, self.max_steps = Path(program), seed, tuple(screen), max_steps
        if not (self.program / 'program.json').is_file():
            raise ValueError(f'{self.program} is not an nkl bundle: it holds no program.json.')
        self.needs = program_needs(self.program)
        self._acts = NodeKitActs(self.needs, self.program.name)
        self.folder = ready_folder(library(), f'NodeKit {self.program.name}', folder)
        self.log_directory = Path(log_directory) if log_directory else self.folder / 'logs'
        self._child, self.steps, self.t_ms, self.result, self.invalid_steps = None, 0, 0.0, None, 0

    def action_spec(self):
        return self._acts

    def instruction(self) -> str:
        acts = '; '.join(f'{name}({", ".join(fields)})' for name, fields in self._acts.describe()['acts'].items())
        return (f'You take part in a task on a screen of {self.screen[0]} x {self.screen[1]} pixels; you see the '
                f'screen as a picture. Act on motor with a list of acts, each {{"act": name, fields}}: {acts}. '
                f'Positions are scene units: the origin at the screen centre, x to the right, y up, one unit one '
                f'pixel. Time passes only by your acts: Wait(ms) lets time pass.')

    def _row(self, answer: dict):
        self.t_ms = answer['t_ms']
        if answer['ended']:
            return {}
        picture = np.frombuffer(base64.b64decode(answer['picture']), np.uint8).reshape(answer['shape'])
        return {'vision:image': picture, 'instruction': self.instruction()}

    def _ended(self, row):
        self.result = self._child.ask({'result': True})
        return row, 'terminal'

    def reset(self, **_):
        self.close()
        home = self.folder.parent / 'nkl-home'      # nkl's data and its Chrome download
        environment = {name: os.environ[name] for name in _RUNTIME_VARIABLES if name in os.environ}
        environment.update(PYTHONNOUSERSITE='1', HOME=str(home), NKL_HOME=str(home / 'nkl'))
        self._child = ChildProcess([environment_python(self.folder), _WORKER, self.program, self.seed, *self.screen],
                                   env=environment, log_directory=self.log_directory,
                                   idle_timeout_s=_CHILD_SILENCE_S).__enter__()
        self.steps, self.result, self.invalid_steps = 0, None, 0
        answer = self._child.ask({'reset': True})
        row = self._row(answer)
        return self._ended(row) if answer['ended'] else row

    def step(self, outputs):
        acts = self._acts.validate(outputs['motor'])
        if acts is None:
            self.invalid_steps += 1
        answer = self._child.ask({'step': acts or []})
        self.steps += 1
        row = self._row(answer)
        if answer['ended']:
            return self._ended(row)
        return row, 'truncated' if self.steps >= self.max_steps else None

    def clock(self):
        return {'t_ms': self.t_ms, 'time_source': 'nkl simulation clock', 'time_inferred': False}

    def close(self):
        if self._child is not None:
            self._child.__exit__(None, None, None)
            self._child = None


def play(subject, program, *, outputs=('motor',), **settings):
    """One run of ``program`` (a bundle folder) by ``subject`` through ``environment_session``:
    ``(result, environment, session)``; ``result`` is None when the run did not complete."""
    from brainscore_core.streaming_helpers import environment_session
    environment = NodeKitEnvironment(program, **settings)
    try:
        session = environment_session(environment, list(outputs))
        subject.participate(session)
        return environment.result, environment, session
    finally:
        environment.close()
