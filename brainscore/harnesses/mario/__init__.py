"""Super Mario Bros. (NES) as a closed-loop environment: a subject plays a level of the CNeuroMod mario data set.

The emulator (stable-retro) runs in a child process with its own environment (``_environment_folder``):
the checkout is CNeuroMod's game integration ``mario.stimuli`` at a pinned commit (the level states the
participants played, ``data.json``, ``scenario.json``, ``rom.sha``), never its ROM. The ROM is a file the
user owns and names (``rom=``); Brain-Score never downloads, copies or stores it: the worker links it into a
temporary integration folder for the run and deletes the folder after. stable-retro checks it against
``rom.sha``. The level states are annexed files of ``mario.stimuli``: ``level_state`` fetches the one
asked for from the public CNeuroMod store with DataLad (the CNeuroMod data plugin's download), at the
same commit.

Each step: the subject's action (one of the 12 button sets of ``ACTIONS``) is held for 4 frames
(60 frames per second); the next row holds those 4 frames on ``vision:video`` ((4, 224, 240, 3) uint8)
and their sound on ``audio`` (mono float32 waveform, the emulator's rate in ``sound_rate_hz``). The play
ends as the scenario of the data set ends it (lives at -1, or the next stage), or after ``max_steps``.

``play_level`` gives the progress with the measures of the participants' replays (``mario.replays``):
``x_traveled`` (scroll position at the end minus at the start) and ``cleared`` (the end state, as
``_check_level_cleared`` of mario.replays).
"""
import base64
import os
from pathlib import Path
from typing import Optional

import numpy as np

from brainscore.harnesses._child_process import ChildProcess
from brainscore.harnesses._environment_folder import Library, _RUNTIME_VARIABLES, environment_python, ready_folder
from brainscore_core.environment import DiscreteSpec

MARIO = Library(
    name='mario', repository='https://github.com/courtois-neuromod/mario.stimuli.git',
    commit='e3ecccf76d7776733ac1a98cf997c97e97f65f66',
    requirements=Path(__file__).with_name('requirements.txt'),
    required_files=('SuperMarioBros-Nes/data.json', 'SuperMarioBros-Nes/scenario.json',
                    'SuperMarioBros-Nes/rom.sha'))
# The NES buttons in stable-retro's order, and the 12 button sets of the trained agents (COMPLEX_MOVEMENT).
NES_BUTTONS = ('B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A')
ACTIONS = ((), ('RIGHT',), ('RIGHT', 'A'), ('RIGHT', 'B'), ('RIGHT', 'A', 'B'), ('A',), ('LEFT',), ('LEFT', 'A'),
           ('LEFT', 'B'), ('LEFT', 'A', 'B'), ('DOWN',), ('UP',))
FRAMES_PER_STEP = 4
INSTRUCTION = 'Play Super Mario Bros.: reach the flag at the end of the level.'
_WORKER = Path(__file__).with_name('_worker.py')


def level_state(level: str) -> Path:
    """The state file of ``level`` at the pinned commit, fetched once into the CNeuroMod download folder."""
    from brainscore.data.cneuromod.data_packaging import extraction, shared
    from brainscore.data.local import data_home
    shared.check_prerequisites()
    clone = shared.clone(data_home() / 'sources' / 'cneuromod', 'mario.stimuli')
    extraction.checkout(clone, MARIO.commit)
    state = clone / 'SuperMarioBros-Nes' / f'{level}.state'
    if not state.is_symlink() and not state.exists():
        raise ValueError(f'{level} is not a level of the integration; the levels are the .state files of '
                         f'{state.parent} (Level1-1, ...).')
    shared.get([state])
    return state


def buttons(action: int) -> list:
    held = ACTIONS[action]
    return [int(name in held) for name in NES_BUTTONS]


def cleared(y_screen: int, lives: int, player_state: int) -> bool:
    """mario.replays' rule on the last frame: not fallen below the screen, not out of lives, not dying."""
    return not (y_screen > 1 or lives == -1 or player_state in (6, 11))


class MarioEnvironment:
    """One play of ``level`` (a state of the integration, ``Level1-1``) with the ROM at ``rom``."""

    def __init__(self, rom, level: str = 'Level1-1', *, max_steps: int = 3000, folder: Optional[Path] = None,
                 log_directory: Optional[Path] = None):
        self.rom, self.level, self.max_steps = Path(rom), level, max_steps
        if not self.rom.is_file():
            raise FileNotFoundError(f'no ROM at {self.rom}: name the Super Mario Bros. (NES) ROM file you own; '
                                    'Brain-Score does not provide it.')
        self.folder = ready_folder(MARIO, f'mario {level}', folder)
        self.state = level_state(level)
        self.log_directory = Path(log_directory) if log_directory else self.folder / 'logs'
        self._child, self.steps, self.trace = None, 0, []

    def action_spec(self):
        return DiscreteSpec(len(ACTIONS))

    def _row(self, answer: dict) -> dict:
        frames = np.frombuffer(base64.b64decode(answer['frames']), np.uint8).reshape(answer['shape'])
        sound = np.frombuffer(base64.b64decode(answer['sound']), np.int16).astype(np.float32) / 32768
        self.sound_rate_hz = answer['sound_rate']
        self.trace.append({key: answer[key] for key in ('scroll', 'lives', 'y_screen', 'player_state', 'done')})
        return {'vision:video': frames, 'audio': sound, 'instruction': INSTRUCTION}

    def reset(self, **_):
        self.close()
        environment = {name: os.environ[name] for name in _RUNTIME_VARIABLES if name in os.environ}
        self._child = ChildProcess([environment_python(self.folder), _WORKER, self.folder / 'checkout', self.rom,
                                    self.state],
                                   env={**environment, 'PYTHONNOUSERSITE': '1'}, log_directory=self.log_directory,
                                   idle_timeout_s=120).__enter__()
        self.steps, self.trace = 0, []
        row = self._row(self._child.ask({'reset': self.level}))
        row['vision:video'] = np.repeat(row['vision:video'], FRAMES_PER_STEP, axis=0)   # the first frame, 4 times
        return row

    def step(self, outputs):
        action = self.action_spec().validate(outputs['motor'])
        row = self._row(self._child.ask({'step': buttons(action), 'frames': FRAMES_PER_STEP}))
        self.steps += 1
        if self.trace[-1]['done']:
            return row, 'terminal'
        return row, 'truncated' if self.steps >= self.max_steps else None

    def clock(self):
        return {'t_ms': self.steps * FRAMES_PER_STEP * 1000 / 60, 'time_source': 'emulator frames',
                'time_inferred': False}

    def progress(self) -> dict:
        """The participants' measures of this play, from its trace."""
        first, last = self.trace[0], self.trace[-1]
        return {'level': self.level, 'steps': self.steps, 'seconds': self.steps * FRAMES_PER_STEP / 60,
                'x_traveled': last['scroll'] - first['scroll'], 'max_scroll': max(t['scroll'] for t in self.trace),
                'cleared': bool(last['done'] and cleared(last['y_screen'], last['lives'], last['player_state'])),
                'ended_by': 'scenario' if last['done'] else 'max_steps'}

    def close(self):
        if self._child is not None:
            self._child.__exit__(None, None, None)
            self._child = None


def play_level(subject, rom, level: str = 'Level1-1', *, outputs=('motor',), max_steps: int = 3000,
               folder: Optional[Path] = None):
    """One play of ``level`` by ``subject`` through ``environment_session``: ``(progress, session)``.
    ``outputs`` may add a region (``neural:cortex``): its recording during play is ``session.collect``."""
    from brainscore_core.streaming_helpers import environment_session
    environment = MarioEnvironment(rom, level, max_steps=max_steps, folder=folder)
    try:
        session = environment_session(environment, list(outputs))
        subject.participate(session)
        return environment.progress(), session
    finally:
        environment.close()
