"""One nkl program (a bundle's ``program.json``) as a simulated participant, in nkl's own environment.

    <environment python> _worker.py <bundle folder> <seed> <screen width> <screen height>

Standard library and nkl only. Commands, one JSON line each: ``{"reset": true}``, ``{"step": [act, ...]}``
(each act ``{"act": <a class name of nkl.sim.actions>, <its fields>}``) and ``{"result": true}``. A reset or
a step answers ``{"done": screen}``: the picture of the screen after the acts (base64 RGB, top row first),
its size, the run's time in ms and whether the run has ended. A result answers the program's result as
value JSON (the form nkl writes in a trace), or None when the run has not completed.

Limit: nkl 0.1.0 starts a run from a ``Call`` of Python code; a bundle holds the lowered program
instead. ``BundleCall`` gives the program, its arguments and its media as a ``Call`` does, and the
worker lets ``nkl.sim`` take it (``op_call``) and finds the bundle's media files (``nkl.media._files``).
Both are names inside nkl 0.1.0; another nkl version must be checked again.
"""
import base64
import json
import os
import sys
import traceback
from pathlib import Path

read_fd, write_fd = (int(fd) for fd in os.environ['BRAINSCORE_CHILD_FDS'].split(','))
commands, messages = os.fdopen(read_fd, 'rb'), os.fdopen(write_fd, 'wb')


def send(message):
    messages.write((json.dumps(message) + '\n').encode())
    messages.flush()


class _Hashed:
    def __init__(self, sha256):
        self.sha256 = sha256


class BundleCall:
    """A bundle's program and arguments in the form ``nkl.sim.Simulation`` reads from a ``Call``."""

    def __init__(self, folder: Path):
        import nkl.media
        document = json.loads((folder / 'program.json').read_text())
        self._program, self._arguments = document['program'], document['arguments']
        sidecar = document.get('sidecar', {})
        self._media = [_Hashed(sha) for sha in sidecar.get('media', {})]
        self._fonts = [_Hashed(sha) for sha in sidecar.get('fonts', {})]
        for kind in ('media', 'fonts'):
            for sha, entry in sidecar.get(kind, {}).items():
                nkl.media._files[sha] = (folder / entry['url']).resolve()

    def program(self):
        return self._program

    def arguments(self):
        return self._arguments

    def media(self):
        return self._media

    def fonts(self):
        return self._fonts

    def decode(self, value):
        return value            # value JSON, as a trace holds it


def screen(simulation):
    ended = simulation.done()
    picture = None if ended else simulation.look()
    return {'picture': base64.b64encode(picture.data).decode() if picture else None,
            'shape': [picture.height, picture.width, 3] if picture else None,
            't_ms': simulation.time_ms(), 'ended': ended}


def main():
    import nkl._core.frontend.program as program
    from nkl.sim import Display, Simulation, actions
    folder, seed, width, height = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    call = BundleCall(folder)
    op_call = program.op_call
    program.op_call = lambda value, use: value if value is call else op_call(value, use)
    simulation = Simulation(call, seed=seed, display=Display(screen=(width, height)), channels='rgb')
    while line := commands.readline():
        command = json.loads(line)
        try:
            if 'reset' in command:
                simulation.reset()
                send({'done': screen(simulation)})
            elif 'step' in command:
                acts = []
                for act in command['step']:
                    fields = {key: value for key, value in act.items() if key != 'act'}
                    if 'points' in fields:
                        fields['points'] = tuple(tuple(point) for point in fields['points'])
                    acts.append(getattr(actions, act['act'])(**fields))
                simulation.step_many(acts)
                send({'done': screen(simulation)})
            else:
                send({'done': simulation.result() if simulation.done()
                      and simulation.outcome().get('kind') == 'completed' else None})
        except Exception as error:
            send({'failed': {'type': type(error).__name__, 'message': str(error),
                             'traceback': traceback.format_exc()}})


if __name__ == '__main__':
    main()
