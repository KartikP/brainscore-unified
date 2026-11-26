"""Any NodeKit (nkl) program as a closed-loop environment: a subject takes part as a person does, with the
keyboard or the pointer, and sees only the screen.

A program is an nkl bundle folder (its ``program.json``: the lowered program, its arguments and the
files of its media). nkl runs it in a child process with its own environment (``_environment_folder``,
Python 3.13): the checkout is NodeKit's task repository at a pinned commit, which carries the nkl wheel
and the bundles. nkl drives its own headless Chrome for Testing (downloaded by nkl at the first run).
Nothing of nkl is imported here.

A step is one turn of nkl's own agent harness (``nkl.agents``), so a run here and a run there with the
same acts have the same acts and outcome in their traces (ours holds the frame looks as well). The row of a
step holds the screen on ``vision:image`` ((H, W, 3) uint8, 1024 x 768 by default), the frames since the
last turn on ``vision:video`` ((T, 384, 512, 3) uint8: one at each frame time the clock passed, 60 frames a
second, at half the screen size, as a participant's screen showed them; their times are
``observation['frame_t_ms']``; absent when the clock passed no frame time; only for a subject that takes
``vision:video``, so for another the trace is nkl's agent harness's line for line), nkl's briefing for the task on ``instruction`` (``system_prompt``: the screen, how
acts take time and how the clock runs under the pacing, the limits, then the benchmark's instructions) and
the facts of the turn on ``observation`` (``task``: the harness's own words that end nkl's briefing, the act
form and the benchmark's instructions, for a subject with its own trained action format; ``t_ms``: the screen's time; ``pointer`` in pixels,
``pointer_down``, ``keys_down``: the subject's own hands; ``turns_left``; ``acts_per_step``; ``act_form``:
the channel the run reads). The subject answers on the act form the benchmark asks for (``form``,
Contract Rules 5.15), ``motor:key`` or ``motor:pointer``, with a list of acts (core's ``MOTOR_ACTS``,
which also checks them): key acts ``press``, ``release`` (a W3C ``KeyboardEvent.code``, such as ``Space``
or ``ArrowLeft``) and ``type``; pointer acts ``move``, ``down``, ``up`` and ``stroke``; ``wait`` on both.
Positions are pixels of the last picture: the origin at the top left, x to the right, y down; the child
converts them with nkl's own conversion. The acts of a step are one schedule: a key or button act takes 1
ms, a stroke as nkl's motor allows, a wait its time; then the subject sees the screen again.

The pacing (``PACINGS``, nkl's): ``frozen`` (the default) stops the task's clock while the subject
answers; ``realtime`` lets the subject's answering time (the wall time from the screen to the acts) pass
before its first act; ``dense`` moves the clock at least ``dense_step_ms`` each step (by default ``DENSE_STEP_MS``, nkl's: one frame;
nkl's own ``act`` timing): a fixed answering time that does not depend on the machine.

The view (``VIEWS``): ``participant`` (the default) is what a participant saw: the frames since the last step
on ``vision:video`` and the screen; ``acts_per_step`` is 20 under every pacing (nkl's own Dense harness allows
one action a turn, so a key press and its release would need two turns). ``agent`` is what nkl's agent harness
gives its agents, under nkl's own rules: one picture a turn and nkl's ``look`` act (``{'act': 'look', 'id'}``,
a picture during the acts); the next row carries ``vision:image`` as named pictures (``screen`` and
``look:<id>`` for each look nkl keeps, ``Limits.looks``), their times in ``observation['looks']``, no frames;
``acts_per_step`` is nkl's (one a turn under dense). A caller may state another ``acts_per_step``. The score
states the view, the pacing, the dense step and the action limit. An
empty list is a valid step with no act and None an answer that could not be read (counted in
``invalid_steps``); under frozen or realtime pacing neither moves the clock. ``MAX_INVALID_IN_A_ROW`` answers
in a row with no act stop the run, as do ``MAX_SAME_IN_A_ROW`` identical act lists in a row (a subject that
is stuck), and a cycle of up to ``CYCLE_PERIODS`` steps of the same acts and screens repeated ``CYCLE_REPEATS``
times (a subject that goes round in a loop; checked not to cut the finished Gabor runs) (``stop_cause``). The run ends when the program
ends (``terminal``; ``result`` is then the program's result as value JSON); when its simulated time passes
``max_ms`` or its steps pass ``max_steps``, the harness stops it as nkl's agent harness does, and it ends
``truncated`` with no result; an nkl error stops the session with the error. Times in the result (response
times) are simulated times.

Each run's nkl trace is written to ``<record>/<folder>/submissions/<name>.json``, the form of a
participant's submission in the task repository, so ``python -m brainscore.harnesses.nodekit replay
<record>`` opens the repository's replay viewer on it (``replay``), as on the participants' runs.

Limit: a subject that takes ``vision:image`` and not ``vision:video`` sees only the screen at the end of each
step: a brief stimulus (the 45 ms Gabor patch) is seen only when a step ends while it is shown. nkl's
``look`` act is not offered (it is not a motor act).

Limit: a session takes its acts on one channel: a program that needs key and pointer acts in one run (type
a text, then click) runs with the one form its benchmark asks for, and a step that needs the other form
cannot be taken.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from brainscore_core.streaming import NamedStreams
from brainscore.harnesses._child_process import ChildProcess
from brainscore.harnesses._environment_folder import Library, _RUNTIME_VARIABLES, environment_python, ready_folder

# NodeKit's task repository: the nkl wheel (vendor/), the bundles (bundles/) and the participants' trials
# (prolific/processed/). Private: the user needs read access. Nothing of it is copied into Brain-Score.
REPOSITORY = 'https://github.com/intelligence-observatory/umi-nkl.git'
COMMIT = '46ff367543ed5a209147ad9b8e6275ce81059d38'
NKL_WHEEL = 'vendor/nkl-0.1.0-py3-none-any.whl'
SCREEN = (1024, 768)
PACINGS = ('frozen', 'realtime', 'dense')
VIEWS = ('participant', 'agent')
PARTICIPANT_ACTS_PER_STEP = 20  # the acts of one step in the participant view, under every pacing
MAX_INVALID_IN_A_ROW = 20       # a run with this many answers in a row that give no act stops (truncated)
MAX_SAME_IN_A_ROW = 40          # a run with this many identical act lists in a row (stuck) stops (truncated)
CYCLE_PERIODS, CYCLE_REPEATS = 10, 5   # a run whose last steps repeat one cycle of (acts, screen) stops (stuck)


def repeats_a_cycle(steps: list, periods: int = CYCLE_PERIODS, repeats: int = CYCLE_REPEATS) -> bool:
    """Whether the last ``repeats`` x P steps (each a pair of the acts and the screen it led to) are one cycle of
    P steps repeated, for a period P up to ``periods``; a cycle of answers with no act does not count (the
    invalid-answer rule stops those)."""
    for period in range(1, periods + 1):
        tail = steps[-repeats * period:]
        if len(tail) == repeats * period and all(tail[i] == tail[i % period] for i in range(len(tail))) \
                and any(acts not in (None, 'null') for acts, _ in tail[:period]):
            return True
    return False
DENSE_STEP_MS = 16.667          # nkl's Dense step: one frame at 60 frames a second
VIDEO = (512, 384)              # the size of a vision:video frame; frames at 60 a second (the worker's FRAME_US)
# The participants' folder of each bundle in the task repository: the replay viewer's names.
VIEWER_FOLDERS = {'drifting-gabor': 'gabor', 'triplet-similarity': 'triplets', 'face-arrangement': 'faces'}
FORMS = ('motor:key', 'motor:pointer')
# What the briefing adds before the benchmark's instructions: the one act form the run reads, and the
# act it does not offer.
_SESSION_WORDS = {'motor:key': 'This task takes keyboard actions only: a pointer action is not done.',
                  'motor:pointer': 'This task takes pointer actions only: a keyboard action is not done.'}
_NO_LOOK = 'The look action is not offered: you see the screen when each reply has played out.'
_WORKER = Path(__file__).with_name('_worker.py')
# Seconds the child may take for one command (a first reset starts nkl's server and Chrome).
_CHILD_SILENCE_S = 600.0


# Chrome's flags for a hardware WebGL2 on Linux: headless Chrome there renders with SwiftShader, which nkl's
# runner refuses ("No GPU: WebGL2"); with these it renders on the GPU through Vulkan.
_LINUX_GPU_FLAGS = ('--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist')


def gpu_chrome(chrome: Path) -> Path:
    """Make nkl's Chrome ``chrome`` start with ``_LINUX_GPU_FLAGS``: the program moves to ``chrome.real`` and
    ``chrome`` becomes a script that adds the flags (once; a second call changes nothing).

    Limit: nkl 0.1.0 starts Chrome with no way to add flags, so the harness wraps the Chrome nkl downloaded
    (``nkl._core.server.chrome.fetch_chrome``); another nkl version must be checked again."""
    real = chrome.with_name('chrome.real')
    if not real.exists():
        chrome.rename(real)
        chrome.write_text(f'#!/bin/sh\nexec "{real}" {" ".join(_LINUX_GPU_FLAGS)} "$@"\n')
        chrome.chmod(0o755)
    return chrome


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


class NodeKitEnvironment:
    """One run of the nkl program in the bundle folder ``program``, with ``seed`` (a bundle's trials are
    already dealt; the seed is nkl's run seed), on a screen of ``screen`` pixels; the subject acts on
    ``form`` (``motor:key`` or ``motor:pointer``) under ``pacing``; ``instructions`` end nkl's briefing."""

    def __init__(self, program, *, form: str, pacing: str = 'frozen', seed: int = 0, screen=SCREEN,
                 max_ms: float = 3_600_000, max_steps: int = 20000, instructions: str = '',
                 dense_step_ms: float = DENSE_STEP_MS, view: str = 'participant', acts_per_step: Optional[int] = None,
                 looks_kept: Optional[int] = None,
                 frames: bool = False, record: Optional[Path] = None, name: str = 'run',
                 folder: Optional[Path] = None, log_directory: Optional[Path] = None):
        self.program, self.seed, self.screen = Path(program), seed, tuple(screen)
        self.max_ms, self.max_steps, self.record, self.name = max_ms, max_steps, record, name
        if not (self.program / 'program.json').is_file():
            raise ValueError(f'{self.program} is not an nkl bundle: it holds no program.json.')
        if form not in FORMS:
            raise ValueError(f'A NodeKit subject acts on {sorted(FORMS)}, not {form!r} (Contract Rules 5.15).')
        if pacing not in PACINGS:
            raise ValueError(f'pacing is one of {PACINGS} (nkl\'s Frozen, Realtime and Dense), not {pacing!r}.')
        if not (isinstance(dense_step_ms, (int, float)) and not isinstance(dense_step_ms, bool) and dense_step_ms > 0):
            raise ValueError(f'dense_step_ms is the time of one dense step in ms, above 0, not {dense_step_ms!r}.')
        if view not in VIEWS:
            raise ValueError(f'view is one of {VIEWS}, not {view!r}.')
        if acts_per_step is not None and (isinstance(acts_per_step, bool) or not isinstance(acts_per_step, int)
                                          or acts_per_step < 1):
            raise ValueError(f'acts_per_step is the most acts of one step, 1 or more, not {acts_per_step!r}.')
        # The agent view gives one picture a turn and the looks: no frames.
        self.form, self.pacing, self.instructions, self.frames = form, pacing, instructions, frames and view == 'participant'
        self.view = view
        # nkl's Limits.looks (None: nkl's own, 4): a provider that takes fewer pictures a request needs fewer.
        if looks_kept is not None and (isinstance(looks_kept, bool) or not isinstance(looks_kept, int) or looks_kept < 0):
            raise ValueError(f'looks_kept is the looks nkl keeps, 0 or more, not {looks_kept!r}.')
        self.looks_kept = looks_kept
        # The participant view keeps 20 acts a step under every pacing; the agent view nkl's own limits.
        self._acts_per_step = acts_per_step if acts_per_step is not None else (
            PARTICIPANT_ACTS_PER_STEP if view == 'participant' else None)
        self.dense_step_ms = dense_step_ms
        self.folder = ready_folder(library(), f'NodeKit {self.program.name}', folder)
        self.log_directory = Path(log_directory) if log_directory else self.folder / 'logs'
        self._child, self.steps, self.t_ms, self.result, self.invalid_steps, self.trace = None, 0, 0.0, None, 0, None
        self.system, self._shown, self.invalid_in_a_row, self.stop_cause = None, None, 0, None
        self.same_in_a_row, self._last_acts, self._steps_seen = 0, None, []
        self.acts_per_step = None
        self._run_folder = uuid.uuid4().hex[:16]

    def _row(self, answer: dict):
        self.t_ms = answer['t_ms']
        if answer['ended']:
            return {}
        picture = answer['picture']
        observation = {'t_ms': picture['at_ms'], **answer['body'], 'turns_left': self.max_steps - self.steps,
                       'acts_per_step': answer['acts_per_step'], 'act_form': self.form, 'task': self.task,
                       'briefing': self.briefing}

        def decoded(part):
            return np.frombuffer(base64.b64decode(part['data']), np.uint8).reshape(part['shape'])
        observation['view'] = self.view
        screen = decoded(picture)
        if self.view == 'agent':           # the screen and each kept look, by name
            screen = NamedStreams({'screen': screen, **{f'look:{look["id"]}': decoded(look)
                                                        for look in answer.get('looks', ())}})
            observation['looks'] = {look['id']: look['at_ms'] for look in answer.get('looks', ())}
        row = {'vision:image': screen, 'instruction': self.system, 'observation': observation}
        frames = answer.get('frames')
        if frames is not None:
            file = Path(frames['file'])
            if frames['shape'][0]:
                row['vision:video'] = np.fromfile(file, np.uint8).reshape(frames['shape'])
                observation['frame_t_ms'] = frames['at_ms']
            if frames['not_sent']:
                observation['frames_not_sent'] = frames['not_sent']
            file.unlink(missing_ok=True)
        self._shown = time.monotonic()
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
        environment = self.child_environment()
        if sys.platform.startswith('linux'):
            fetched = subprocess.run([str(environment_python(self.folder)), '-c', 'from nkl._core.server.chrome import '
                                      'fetch_chrome; print(fetch_chrome())'], env=environment, capture_output=True,
                                     text=True, check=True)
            gpu_chrome(Path(fetched.stdout.strip().splitlines()[-1]))
        self._child = ChildProcess([environment_python(self.folder), _WORKER, self.program, self.seed, *self.screen,
                                    *VIDEO],
                                   env=environment, log_directory=self.log_directory,
                                   idle_timeout_s=_CHILD_SILENCE_S).__enter__()
        self.steps, self.result, self.invalid_steps, self.trace = 0, None, 0, None
        words = _SESSION_WORDS[self.form] + (f' {_NO_LOOK}' if self.view == 'participant' else '')
        self.task = f'{_SESSION_WORDS[self.form]}\n\n{self.instructions}'.strip()
        # The arguments of nkl's briefing, also in each row (observation['briefing']): a subject that reads
        # positions on its own grid has nkl write the same briefing for its screen.
        self.briefing = {'pacing': [self.pacing.capitalize(), round(self.dense_step_ms * 1000)],
                         'limits': {'turns': self.max_steps, 'actions': self._acts_per_step, 'looks': self.looks_kept},
                         'seconds': self.max_ms / 1000, 'instructions': f'{words}\n\n{self.instructions}'.strip()}
        answer = self._child.ask({'reset': self.briefing})
        self.system, self.acts_per_step = answer['system'], answer['acts_per_step']
        if answer['extent'] != list(self.screen):
            raise RuntimeError(f'nkl shows the program on {answer["extent"]} scene units, not the {self.screen} '
                               'pixels of the screen; the briefing and the act positions assume one unit a pixel.')
        row = self._row(answer)
        return self._ended(row, 'terminal') if answer['ended'] else row

    def step(self, outputs):
        thought = time.monotonic() - self._shown
        acts = outputs[self.form]          # checked by the session (core's MOTOR_ACTS)
        if self.view == 'participant' and any(act['act'] == 'look' for act in acts or ()):
            raise ValueError(f'{self.program.name} runs in the participant view: a look is offered only in the '
                             f'agent view (view="agent"); the frames since the last step are on vision:video.')
        self.invalid_in_a_row = self.invalid_in_a_row + 1 if acts is None else 0
        self.same_in_a_row = self.same_in_a_row + 1 if acts and acts == self._last_acts else 0
        self._last_acts = acts
        if acts is None:
            self.invalid_steps += 1
        frames = None
        if self.frames:
            (self.folder / 'frames').mkdir(exist_ok=True)
            frames = str(self.folder / 'frames' / f'{os.getpid()}-{id(self)}.raw')
        answer = self._child.ask({'step': {'acts': acts or [], 'seconds': thought, 'frames': frames}})
        self.steps += 1
        row = self._row(answer)
        if answer['ended']:
            return self._ended(row, 'terminal')
        if row:
            screen = row['vision:image']
            screen = screen['screen'] if isinstance(screen, dict) else screen
            self._steps_seen = (self._steps_seen + [(json.dumps(acts, sort_keys=True),
                                                     hashlib.sha1(screen.tobytes()).hexdigest())])[-CYCLE_PERIODS * CYCLE_REPEATS:]
        cycle = repeats_a_cycle(self._steps_seen)
        stuck = self.same_in_a_row + 1 >= MAX_SAME_IN_A_ROW or cycle
        if (self.t_ms >= self.max_ms or self.steps >= self.max_steps or self.invalid_in_a_row >= MAX_INVALID_IN_A_ROW
                or stuck):
            self.stop_cause = ('time' if self.t_ms >= self.max_ms else 'steps' if self.steps >= self.max_steps
                               else 'invalid answers in a row' if self.invalid_in_a_row >= MAX_INVALID_IN_A_ROW
                               else 'a cycle of the same acts and screens' if cycle else 'the same acts in a row')
            return self._ended(row, 'truncated')
        return row, None

    def clock(self):
        return {'t_ms': self.t_ms, 'time_source': 'nkl simulation clock', 'time_inferred': False}

    def child_environment(self) -> dict:
        """The child's environment: a home shared by every run (nkl's Chrome download) and nkl's data folder of
        this run alone, so each run starts and owns its nkl server: nkl attaches to a server another process
        started in the same data folder, and that server ends with that process."""
        home = self.folder.parent / 'nkl-home'
        environment = {name: os.environ[name] for name in _RUNTIME_VARIABLES if name in os.environ}
        environment.update(PYTHONNOUSERSITE='1', HOME=str(home),
                           NKL_HOME=str(home / 'runs' / self._run_folder))
        return environment

    def close(self):
        if self._child is not None:
            self._child.__exit__(None, None, None)
            self._child = None
            shutil.rmtree(self.child_environment()['NKL_HOME'], ignore_errors=True)


def play(subject, program, *, form: str, outputs=None, **settings):
    """One run of ``program`` (a bundle folder) by ``subject`` on the act form ``form`` through
    ``environment_session``:
    ``(result, environment, session)``; ``result`` is None when the run did not complete. ``outputs`` are the
    channels the session requests (the act form by default; with ``neural:<region>`` beside it the subject's
    layers for the region are recorded at each turn, ``session.collect``). With ``record=`` the run's trace is
    kept there under the subject's identifier and the time."""
    import time
    from brainscore_core.streaming_helpers import environment_session
    settings.setdefault('name', f"{getattr(subject, 'identifier', 'subject')}-{time.strftime('%Y%m%dT%H%M%S')}")
    # The frames go only to a subject that takes them (a subject's declared inputs), in the participant view.
    settings.setdefault('frames', 'vision:video' in set(getattr(subject, 'in_channels', ())))
    environment = NodeKitEnvironment(program, form=form, **settings)
    try:
        requested = list(outputs or (form,))
        if form not in requested:
            raise ValueError(f'outputs {requested} must hold the act form {form!r} the run reads.')
        # history_limit=0: the source keeps no copy of each screen and act; the session keeps the responses.
        session = environment_session(environment, requested, action_channel=form, history_limit=0)
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
