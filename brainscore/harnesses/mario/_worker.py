"""The Super Mario Bros. emulator in its own environment (stable-retro); the standard library, numpy and retro only.

    <environment python> _worker.py <checkout of mario.stimuli> <ROM file> <level state file>

The game's integration (``data.json``, ``scenario.json``, ``rom.sha``) is read from the checkout, the level
state from its file; the ROM is the file the user gives. A temporary integration folder holds copies of the checkout's
files and a link to the ROM; it is deleted when the worker ends. Commands, one JSON line each:
``{"reset": "<state>"}`` and ``{"step": [9 buttons], "frames": n}``; each answers ``{"done": observation}``:
the frames of the step (base64, uint8, (n, 224, 240, 3)), the sound (base64, int16, mono), the sound rate, the
scroll position, the lives, the player's screen row and state, and whether the scenario ended the play.
"""
import base64
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

GAME = 'SuperMarioBros-Nes'
read_fd, write_fd = (int(fd) for fd in os.environ['BRAINSCORE_CHILD_FDS'].split(','))
commands, messages = os.fdopen(read_fd, 'rb'), os.fdopen(write_fd, 'wb')


def send(message):
    messages.write((json.dumps(message) + '\n').encode())
    messages.flush()


def observation(env, frames, sound, done):
    data = env.data.lookup_all()
    mono = np.asarray(sound, dtype=np.int16).reshape(len(sound), -1).mean(axis=1).astype(np.int16) if len(sound) \
        else np.zeros(0, np.int16)
    return {'frames': base64.b64encode(np.ascontiguousarray(np.stack(frames), dtype=np.uint8).tobytes()).decode(),
            'shape': list(np.stack(frames).shape), 'sound': base64.b64encode(mono.tobytes()).decode(),
            'sound_rate': float(env.em.get_audio_rate()),
            # the variables of the participants' measures (mario.replays): scroll position and the end state
            'scroll': int(data['xscrollHi']) * 256 + int(data['xscrollLo']), 'lives': int(data['lives']),
            'y_screen': int(data['player_y_screen']), 'player_state': int(data['player_state']), 'done': bool(done)}


def main():
    checkout, rom, state = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    import stable_retro as retro
    folder = Path(tempfile.mkdtemp(prefix='mario-integration-'))
    env = None
    try:
        game = folder / GAME
        game.mkdir()
        for name in ('data.json', 'scenario.json', 'metadata.json', 'rom.sha'):
            if (checkout / GAME / name).is_file():
                shutil.copyfile(checkout / GAME / name, game / name)
        shutil.copyfile(state, game / state.name)
        (game / 'rom.nes').symlink_to(rom.resolve())
        retro.data.Integrations.add_custom_path(str(folder))
        while line := commands.readline():
            command = json.loads(line)
            try:
                if 'reset' in command:
                    if env is not None:
                        env.close()
                    env = retro.make(GAME, state=command['reset'], inttype=retro.data.Integrations.CUSTOM_ONLY,
                                     render_mode=None)
                    frame, _ = env.reset()
                    send({'done': observation(env, [frame], env.em.get_audio(), False)})
                else:
                    frames, sound, done = [], [], False
                    for _ in range(command['frames']):
                        frame, _, terminated, truncated, _ = env.step(np.asarray(command['step'], dtype=np.int8))
                        frames.append(frame)
                        sound.extend(env.em.get_audio().tolist())
                        done = bool(terminated or truncated)
                        if done:
                            break
                    send({'done': observation(env, frames, sound, done)})
            except Exception as error:
                send({'failed': {'type': type(error).__name__, 'message': str(error),
                                 'traceback': traceback.format_exc()}})
    finally:
        if env is not None:
            env.close()
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == '__main__':
    main()
