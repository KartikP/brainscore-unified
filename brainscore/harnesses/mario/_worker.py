"""A stable-retro game in its own environment; the standard library, numpy and stable_retro only.

    <environment python> _worker.py <game> <state> [<integration checkout> <ROM file> <state file>]

With the three last arguments (Super Mario Bros.), a temporary integration folder holds copies of the
checkout's ``data.json``, ``scenario.json``, ``metadata.json``, ``rom.sha`` and the state file, and a link to
the ROM; it is deleted when the worker ends. Without them, the game is one stable-retro ships (with its ROM).
Commands, one JSON line each: ``{"reset": true}`` and ``{"step": [button names], "frames": n}``; each answers
``{"done": observation}``: the frames (base64, uint8, (n, H, W, 3)), the sound (base64, int16, mono), the sound
rate, the game's variables (``data.json``) and whether the scenario ended the play.
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

read_fd, write_fd = (int(fd) for fd in os.environ['BRAINSCORE_CHILD_FDS'].split(','))
commands, messages = os.fdopen(read_fd, 'rb'), os.fdopen(write_fd, 'wb')


def send(message):
    messages.write((json.dumps(message) + '\n').encode())
    messages.flush()


def observation(env, frames, sound, done):
    sound = np.asarray(sound, dtype=np.int16)
    mono = sound.reshape(len(sound), -1).mean(axis=1).astype(np.int16) if len(sound) else np.zeros(0, np.int16)
    stack = np.ascontiguousarray(np.stack(frames), dtype=np.uint8)
    return {'frames': base64.b64encode(stack.tobytes()).decode(), 'shape': list(stack.shape),
            'sound': base64.b64encode(mono.tobytes()).decode(), 'sound_rate': float(env.em.get_audio_rate()),
            'variables': {key: int(value) for key, value in env.data.lookup_all().items()}, 'done': bool(done)}


def main():
    import stable_retro as retro
    game, state = sys.argv[1], sys.argv[2]
    folder, env, inttype = None, None, retro.data.Integrations.STABLE
    try:
        if len(sys.argv) > 3:
            checkout, rom, state_file = (Path(argument) for argument in sys.argv[3:6])
            folder = Path(tempfile.mkdtemp(prefix='retro-integration-'))
            (folder / game).mkdir()
            for name in ('data.json', 'scenario.json', 'metadata.json', 'rom.sha'):
                if (checkout / game / name).is_file():
                    shutil.copyfile(checkout / game / name, folder / game / name)
            shutil.copyfile(state_file, folder / game / state_file.name)
            (folder / game / 'rom.nes').symlink_to(rom.resolve())
            retro.data.Integrations.add_custom_path(str(folder))
            inttype = retro.data.Integrations.CUSTOM_ONLY
        while line := commands.readline():
            command = json.loads(line)
            try:
                if 'reset' in command:
                    if env is not None:
                        env.close()
                    env = retro.make(game, state=state, inttype=inttype, render_mode=None)
                    frame, _ = env.reset()
                    send({'done': observation(env, [frame], env.em.get_audio(), False)})
                    continue
                held = np.array([int(name in command['step']) for name in env.buttons], dtype=np.int8)
                frames, sound, done = [], [], False
                for _ in range(command['frames']):
                    frame, _, terminated, truncated, _ = env.step(held)
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
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == '__main__':
    main()
