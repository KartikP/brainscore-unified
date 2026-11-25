"""One nkl program (a bundle's ``program.json``) as a simulated participant, in nkl's own environment.

    <environment python> _worker.py <bundle folder> <seed> <screen width> <screen height> <frame ms> <video width> <video height>

Standard library and nkl only. Commands, one JSON line each: ``{"reset": true}``, ``{"step": [act, ...]}``
(each act ``{"act": <a class name of nkl.sim.actions>, <its fields>}``), ``{"leave": true}``,
``{"result": true}`` and ``{"trace": <file>}``. A reset or a step answers ``{"done": screen}``: the frames
since the last step (one at each frame time the clock passed, at the video size), the picture of the
screen at the end at full size (base64 RGB, top row first), the run's time in ms and whether the run has
ended. A step's time runs to at least the next frame time: when its acts take less, a wait is added, so
every step (also one with no act) gives one frame or more. A result answers the program's result as value
JSON (the form nkl writes in a trace), or None when the run has not completed; a trace writes the run's
trace lines to the file as one JSON array (the form of a participant's submission).

Limit: a frame is taken after an act, not during it: the frames of a pointer stroke that spans several
frame times are one frame at its end.

Limit: nkl 0.1.0 starts a run from a ``Call`` of Python code; a bundle holds the lowered program
instead. ``BundleCall`` gives the program, its arguments and its media as a ``Call`` does, and the
worker lets ``nkl.sim`` take it (``op_call``) and finds the bundle's media files (``nkl.media._files``).
Both are names inside nkl 0.1.0; another nkl version must be checked again.
"""
import base64
import json
import math
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


def encoded(pictures):
    return {'data': base64.b64encode(b''.join(p.data for p in pictures)).decode(),
            'shape': [len(pictures), pictures[0].height, pictures[0].width, 3]} if pictures else None


def screen(simulation, frames):
    ended = simulation.done()
    picture = None if ended else simulation.look()
    return {'frames': encoded(frames), 'picture': encoded([picture]) if picture else None,
            't_ms': simulation.time_ms(), 'ended': ended}


def duration_ms(act, simulation, pointer):
    """How long nkl takes for ``act`` (``Simulation.step_many``): a key or button act the motor's press time,
    a typed text its presses, a stroke as its motor allows; a move none. ``pointer`` is (x, y) before it."""
    from nkl.sim import Motor, actions, stroke_samples
    press = Motor().press_ms
    if isinstance(act, (actions.KeyPress, actions.KeyRelease, actions.PointerDown, actions.PointerUp)):
        return press
    if isinstance(act, actions.TypeText):
        return sum(4 if actions.TYPED[c][1] else 2 for c in act.text) * press
    if isinstance(act, actions.PointerStroke):
        return stroke_samples(pointer, act, Motor())[1] / 1000
    return 0.0


def next_frame(now, frame_ms):
    """The first frame time after ``now``; nkl's clock counts whole microseconds, so a time within 2 us of a
    frame time is that time."""
    following = (math.floor(now / frame_ms) + 1) * frame_ms
    return following + frame_ms if following - now < 0.002 else following


def run(simulation, plan, looks_per_request=20):
    """The plan in requests of at most ``looks_per_request`` looks each (nkl's answer to one request holds
    every picture of it); the pictures in order."""
    from nkl.sim import actions
    pictures, part = [], []
    for item in plan:
        part.append(item)
        if isinstance(item, actions.Look) and sum(isinstance(i, actions.Look) for i in part) >= looks_per_request:
            pictures += simulation.step_many(part)
            part = []
    return pictures + (simulation.step_many(part) if part else [])


def planned(acts, simulation, frame_ms, video):
    """The acts with a look at each frame time the clock passes, and a wait to the next frame time when
    the acts end before it."""
    from nkl.sim import actions
    now = simulation.time_ms()
    following = next_frame(now, frame_ms)
    plan, looks = [], 0
    state = simulation.pointer()
    pointer = (state.x, state.y)

    def wait(ms):
        nonlocal now, following, looks
        end = now + ms
        while following <= end + 1e-9:
            plan.extend([actions.Wait(ms=following - now), actions.Look(size=video)])
            now, following, looks = following, following + frame_ms, looks + 1
        if end > now:
            plan.append(actions.Wait(ms=end - now))
            now = end
    for act in acts:
        if isinstance(act, actions.Wait):
            wait(act.ms)
            continue
        now += duration_ms(act, simulation, pointer)
        plan.append(act)
        if isinstance(act, (actions.PointerMove, actions.PointerStroke)):
            pointer = (act.x, act.y) if isinstance(act, actions.PointerMove) else act.points[-1]
        if now >= following - 1e-9:
            plan.append(actions.Look(size=video))
            following, looks = next_frame(now, frame_ms), looks + 1
    if not looks:
        wait(following - now)
    return plan


def main():
    import nkl._core.frontend.program as program
    from nkl.sim import Display, Simulation, actions
    folder, seed, width, height = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    frame_ms, video = float(sys.argv[5]), (int(sys.argv[6]), int(sys.argv[7]))
    call = BundleCall(folder)
    op_call = program.op_call
    program.op_call = lambda value, use: value if value is call else op_call(value, use)
    simulation = Simulation(call, seed=seed, display=Display(screen=(width, height)), channels='rgb')
    while line := commands.readline():
        command = json.loads(line)
        try:
            if 'reset' in command:
                simulation.reset()
                send({'done': screen(simulation, [])})
            elif 'step' in command:
                acts = []
                for act in command['step']:
                    fields = {key: value for key, value in act.items() if key != 'act'}
                    if 'points' in fields:
                        fields['points'] = tuple(tuple(point) for point in fields['points'])
                    acts.append(getattr(actions, act['act'])(**fields))
                frames = run(simulation, planned(acts, simulation, frame_ms, video))
                send({'done': screen(simulation, frames)})
            elif 'leave' in command:
                if not simulation.done():
                    simulation.leave()
                send({'done': True})
            elif 'trace' in command:
                Path(command['trace']).write_text(json.dumps(simulation.trace()))
                send({'done': command['trace']})
            else:
                send({'done': simulation.result() if simulation.done()
                      and simulation.outcome().get('kind') == 'completed' else None})
        except Exception as error:
            send({'failed': {'type': type(error).__name__, 'message': str(error),
                             'traceback': traceback.format_exc()}})


if __name__ == '__main__':
    main()
