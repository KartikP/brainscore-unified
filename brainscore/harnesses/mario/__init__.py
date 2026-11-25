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
GAME = 'SuperMarioBros-Nes'
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


ROM_SHA1 = 'facee9c577a5262dbe33ac4930bb0b58c8c037f7'     # rom.sha of the integration: Super Mario Bros. (NES)


def buttons(action: int) -> list:
    """The NES buttons held by ``action``, in stable-retro's order."""
    held = ACTIONS[action]
    return [int(name in held) for name in NES_BUTTONS]


def cleared(y_screen: int, lives: int, player_state: int) -> bool:
    """mario.replays' rule on the last frame: not fallen below the screen, not out of lives, not dying."""
    return not (y_screen > 1 or lives == -1 or player_state in (6, 11))


def check_rom(rom) -> Path:
    """``rom`` when it is the Super Mario Bros. ROM of the integration (its SHA1); else a refusal before any work."""
    import hashlib
    rom = Path(rom)
    if not rom.is_file():
        raise FileNotFoundError(f'no ROM at {rom}: name the Super Mario Bros. (NES) ROM file you own; '
                                'Brain-Score does not provide it.')
    digest = hashlib.sha1(rom.read_bytes()).hexdigest()
    if digest != ROM_SHA1:
        raise ValueError(f'{rom} has SHA1 {digest}; the data set was recorded with the Super Mario Bros. (NES) ROM '
                         f'of SHA1 {ROM_SHA1} (rom.sha of courtois-neuromod/mario.stimuli). Name that ROM file.')
    return rom


class RetroEnvironment:
    """One play of ``state`` of a stable-retro ``game`` in the child process. ``integration`` is
    ``(checkout, ROM, state file)`` for a game stable-retro does not ship (Super Mario Bros.); None for one it
    ships with its ROM. The subject's action is one of ``ACTIONS`` (button names), held for 4 frames."""

    def __init__(self, game: str, state: str, *, integration=None, max_steps: int = 3000,
                 folder: Optional[Path] = None, log_directory: Optional[Path] = None):
        self.game, self.state, self.integration, self.max_steps = game, state, integration, max_steps
        self.folder = ready_folder(MARIO, f'{game} {state}', folder)
        self.log_directory = Path(log_directory) if log_directory else self.folder / 'logs'
        self._child, self.steps, self.trace = None, 0, []

    def action_spec(self):
        return DiscreteSpec(len(ACTIONS))

    def _row(self, answer: dict) -> dict:
        frames = np.frombuffer(base64.b64decode(answer['frames']), np.uint8).reshape(answer['shape'])
        sound = np.frombuffer(base64.b64decode(answer['sound']), np.int16).astype(np.float32) / 32768
        self.sound_rate_hz = answer['sound_rate']
        self.trace.append({**answer['variables'], 'done': answer['done']})
        return {'vision:video': frames, 'audio': sound, 'instruction': INSTRUCTION}

    def reset(self, **_):
        self.close()
        environment = {name: os.environ[name] for name in _RUNTIME_VARIABLES if name in os.environ}
        extra = [str(part) for part in self.integration] if self.integration else []
        self._child = ChildProcess([environment_python(self.folder), _WORKER, self.game, self.state, *extra],
                                   env={**environment, 'PYTHONNOUSERSITE': '1'}, log_directory=self.log_directory,
                                   idle_timeout_s=120).__enter__()
        self.steps, self.trace = 0, []
        row = self._row(self._child.ask({'reset': True}))
        row['vision:video'] = np.repeat(row['vision:video'], FRAMES_PER_STEP, axis=0)   # the first frame, 4 times
        return row

    def step(self, outputs):
        action = self.action_spec().validate(outputs['motor'])
        row = self._row(self._child.ask({'step': list(ACTIONS[action]), 'frames': FRAMES_PER_STEP}))
        self.steps += 1
        if self.trace[-1]['done']:
            return row, 'terminal'
        return row, 'truncated' if self.steps >= self.max_steps else None

    def clock(self):
        return {'t_ms': self.steps * FRAMES_PER_STEP * 1000 / 60, 'time_source': 'emulator frames',
                'time_inferred': False}

    def close(self):
        if self._child is not None:
            self._child.__exit__(None, None, None)
            self._child = None


class MarioEnvironment(RetroEnvironment):
    """One play of ``level`` (a state of the integration, ``Level1-1``) with the ROM at ``rom`` (checked first)."""

    def __init__(self, rom, level: str = 'Level1-1', **kwargs):
        rom = check_rom(rom)
        super().__init__(GAME, level, max_steps=kwargs.pop('max_steps', 3000), **kwargs)
        self.level = level
        self.integration = (self.folder / 'checkout', rom, level_state(level))

    def progress(self) -> dict:
        """The participants' measures of this play (mario.replays): scroll distance and the clear rule."""
        def scroll(t):
            return t['xscrollHi'] * 256 + t['xscrollLo']
        first, last = self.trace[0], self.trace[-1]
        return {'level': self.level, 'steps': self.steps, 'seconds': self.steps * FRAMES_PER_STEP / 60,
                'x_traveled': scroll(last) - scroll(first), 'max_scroll': max(scroll(t) for t in self.trace),
                'cleared': bool(last['done'] and cleared(last['player_y_screen'], last['lives'], last['player_state'])),
                'ended_by': 'scenario' if last['done'] else 'max_steps'}


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


def participants_progress(level: str = 'Level1-1') -> dict:
    """Per participant, the plays of ``level`` in the scanner (the info files of ``mario.replays``, fetched with
    DataLad): plays, share cleared, median and largest ``x_traveled``, median seconds."""
    import glob
    import json
    import statistics
    from brainscore.data.cneuromod.data_packaging import shared
    from brainscore.data.local import data_home
    shared.check_prerequisites()
    clone = shared.clone(data_home() / 'sources' / 'cneuromod', 'mario.replays')
    code = 'w{}l{}'.format(*level.removeprefix('Level').split('-'))
    files = [Path(path) for path in sorted(glob.glob(str(clone / f'sub-*/ses-*/beh/infos/*_level-{code}_rep-*.json')))]
    shared.get(files)
    plays = [json.loads(path.read_text()) for path in files]
    table = {}
    for participant in sorted({play['Subject'] for play in plays}):
        own = [play for play in plays if play['Subject'] == participant]
        table[f'sub-{participant}'] = {
            'plays': len(own), 'cleared': sum(play['Cleared'] for play in own) / len(own),
            'median_x_traveled': statistics.median(play['X_Traveled'] for play in own),
            'max_x_traveled': max(play['X_Traveled'] for play in own),
            'median_seconds': statistics.median(play['Duration'] for play in own)}
    return table
