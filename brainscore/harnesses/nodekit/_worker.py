"""One nkl program (a bundle's ``program.json``) as a simulated participant, in nkl's own environment.

    <environment python> _worker.py <bundle folder> <seed> <screen width> <screen height> <video width> <video height>

Standard library and nkl only. A step is one turn of nkl's agent harness (``nkl.agents``): the turn's
acts as one schedule (``act``), then a look at the screen, under the run's pacing; the body (pointer, keys)
follows the acts (``observe``). When the step names a frames file, the worker also looks at each frame time
the clock passes (60 frames a second, ``FRAME_US``, at the video size; each look is a frame line in the
trace) and writes the frames there; without one, the trace is nkl's agent harness's line for line. Commands, one JSON line each:

- ``{"reset": {"pacing": [kind, step µs], "limits": {...}, "seconds": s, "instructions": text, "picture": [w, h]
  or null, "motor": {...} or null, "shown": n or null, "main": [argument, ...]}}`` (an action
  limit of None: nkl's own, one a turn under Dense; a picture of None: the screen's size; a motor of None: nkl's
  own; with ``shown``, the bundle arguments ``main`` (the main block's lists of trials) keep their first n) starts
  the run (``begin``) and answers the first screen with nkl's briefing for it (``system_prompt``).
- ``{"step": {"acts": [act, ...], "seconds": s, "frames": file or null}}``: the acts in the form of Brain-Score's ``motor:key`` and
  ``motor:pointer`` (pixels of the screen, origin top left), ``seconds`` the subject's answering time (it
  passes in the run only under the Realtime pacing).
- ``{"leave": true}`` stops a run that is still going, as nkl's agent harness does at its end.
- ``{"result": true}``: the program's result as value JSON, or None when it did not complete.
- ``{"trace": <file>}`` writes the run's trace lines to the file as one JSON array.

A reset or a step answers ``{"done": screen}``: the picture (base64 RGB, top row first), the frames file
with the frames' shape and times and the count not sent, the run's time in ms, whether it has ended, and the body and limits the
subject's briefing shows.

Limit: a step sends at most ``MAX_FRAMES`` frames (10 s); the frame times after that are counted
(``not_sent``), not looked at.

Limit: a frame is a look between two scheduled items: the positions of a pointer stroke between its
samples are not frames of their own (a stroke is sampled at nkl's motor rate, 120 a second).

Limit: nkl 0.1.0 starts a run from a ``Call`` of Python code; a bundle holds the lowered program
instead. ``BundleCall`` gives the program, its arguments and its media as a ``Call`` does, and the
worker lets ``nkl.fsim`` take it (``op_call``) and finds the bundle's media files (``nkl.media._files``).
It converts pixels with nkl's own ``to_scene`` and ``to_frame`` and makes strokes with its ``Stroke``. These are
names inside the nkl of the pinned task repository (``NKL_COMMIT`` site-20261119-230637-272-gd9538ac6ef10);
another nkl version must be checked again.

Limit: ``shown`` cuts the bundle's main-block trial lists; the task code's own ``shown`` (a newer task file) also
keeps the Gabor trial counter at the full count ("3 of 50"), the cut bundle counts the trials kept ("3 of 10").
"""
import base64
import json
import os
import sys
import traceback
from dataclasses import replace
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
    """A bundle's program and arguments in the form ``nkl.fsim.start`` reads from a ``Call``."""

    def __init__(self, folder: Path, shown=None, main=()):
        import nkl.media
        document = json.loads((folder / 'program.json').read_text())
        self._program, self._arguments = document['program'], document['arguments']
        if shown is not None:          # the main block's first trials only: each of its parallel lists is cut
            self._arguments = [value[:shown] if number in main else value
                               for number, value in enumerate(self._arguments)]
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


def start(folder: Path, seed: int, screen, picture=None, shown=None, main=()):
    """A fresh run of the bundle, as ``nkl.fsim.start`` makes one for nkl's agent harness; its looks are
    ``picture`` pixels (None: the screen's)."""
    import nkl._core.frontend.program as program
    from nkl import fsim
    call = BundleCall(folder, shown, main)
    op_call = program.op_call
    program.op_call = lambda value, use: value if value is call else op_call(value, use)
    try:
        return fsim.start(call, seed=seed, display=fsim.Display(screen=screen),
                          look=fsim.Look(size=tuple(picture) if picture else None, channels='rgb'))
    finally:
        program.op_call = op_call


def pacing_of(kind, step):
    from nkl import agents
    return {'Frozen': agents.Frozen(), 'Realtime': agents.Realtime(), 'Dense': agents.Dense(step)}[kind]


def actions(acts, context):
    """nkl's agent actions for Brain-Score acts: pixels become scene units by nkl's ``to_scene``; a typed
    text becomes its key presses as nkl's reply reader makes them."""
    from nkl import agents, fsim
    from nkl._core.host.agents import Stroke, TakeLook, to_scene
    made = []

    def scene(x, y):
        return to_scene(x, y, (context.screen.width, context.screen.height), context.extent)
    for act in acts:
        name = act['act']
        if name == 'press':
            made.append(fsim.Keyboard.Press(act['key']))
        elif name == 'release':
            made.append(fsim.Keyboard.Release(act['key']))
        elif name == 'type':
            typed, _, problems, _ = agents.parse(json.dumps({'do': 'type', 'text': act['text']}), context)
            if problems:
                raise ValueError('; '.join(problems))
            made.extend(typed)
        elif name == 'wait':
            made.append(agents.Wait(round(act['ms'] * 1000)))
        elif name == 'move':
            made.append(fsim.Pointer.Move(*scene(act['x'], act['y'])))
        elif name == 'down':
            made.append(fsim.Pointer.Down())
        elif name == 'up':
            made.append(fsim.Pointer.Up())
        elif name == 'look':
            made.append(TakeLook(act['id']))
        elif name == 'stroke':
            made.append(Stroke(tuple(scene(*point) for point in act['points'])))
        else:
            raise ValueError(f'{name!r} is not an act of motor:key or motor:pointer')
    return made


FRAME_US = 1_000_000 / 60
LOOKS_PER_REQUEST = 20          # nkl's answer to one request holds every picture of it


def frame_time(k):
    return round(k * FRAME_US)


MAX_FRAMES = 600                # the most frames one step sends: 10 s at 60 a second


class Frames:
    """The frames of one step, written as they come to ``file`` (raw RGB, one after the other): at most
    ``MAX_FRAMES``; the frame times after that are counted in ``not_sent`` and not looked at."""

    def __init__(self, file):
        self.file, self.at_ms, self.shape, self.not_sent = open(file, 'wb'), [], None, 0

    def add(self, image):
        self.file.write(image.data)
        self.at_ms.append(image.at / 1000)
        self.shape = [image.height, image.width, 3]

    def answer(self):
        self.file.close()
        return {'file': self.file.name, 'shape': [len(self.at_ms), *(self.shape or [0, 0, 3])],
                'at_ms': self.at_ms, 'not_sent': self.not_sent}


def with_frames(video, frames: Frames):
    """``nkl.fsim.run`` with a look at each frame time the schedule passes (after any item at that time), in
    requests of at most ``LOOKS_PER_REQUEST`` looks, the last with the schedule's ``until``; no request after
    the run ends. The frames go to ``frames``; the schedule's own looks come back."""
    import math
    from nkl import fsim
    real = fsim.run

    def run(r, schedule, until=None):
        t, items, k = r.t, [], math.floor(r.t / FRAME_US) + 1
        while frame_time(k) <= r.t:
            k += 1
        looked = len(frames.at_ms)
        for dt, item in schedule:
            at, previous = t + dt, t
            while frame_time(k) < at:
                if looked < MAX_FRAMES:
                    items.append((frame_time(k) - previous, fsim.Look(size=video), True))
                    previous, looked = frame_time(k), looked + 1
                else:
                    frames.not_sent += 1
                k += 1
            items.append((at - previous, item, False))
            t = at
        own, start = [], 0
        while start < len(items) and r.outcome is None:
            end, count = start, 0
            while end < len(items) and count < LOOKS_PER_REQUEST:
                count += items[end][2]
                end += 1
            part = items[start:end]
            r, images = real(r, [(dt, what) for dt, what, _ in part], until if end == len(items) else None)
            looks = [frame for _, what, frame in part if isinstance(what, fsim.Look)]
            for image, frame in zip(images, looks):
                frames.add(image) if frame else own.append(image)
            start = end
        return r, own
    return run


def screen(run, context, system=None, frames=None):
    from nkl._core.host.agents import to_frame
    ended = run.outcome is not None
    image, body = context.screen, context.body
    x, y = to_frame(body.pointer.x, body.pointer.y, (image.width, image.height), context.extent)
    answer = {'t_ms': run.t / 1000, 'ended': ended,
              'body': {'pointer': [x, y], 'pointer_down': body.pointer.down, 'pointer_connected': body.pointer.connected,
                       'keys_down': list(body.keyboard.down)},
              'extent': list(context.extent), 'acts_per_step': context.limits.actions}
    if not ended:
        answer['picture'] = {'data': base64.b64encode(image.data).decode(), 'shape': [image.height, image.width, 3],
                             'at_ms': image.at / 1000}
    if frames is not None:
        answer['frames'] = frames.answer()
    if context.looks:               # the looks nkl keeps (Limits.looks), oldest first
        answer['looks'] = [{'id': name, 'data': base64.b64encode(look.data).decode(),
                            'shape': [look.height, look.width, 3], 'at_ms': look.at / 1000}
                           for name, look in context.looks]
    if system is not None:
        answer['system'] = system
    return answer


def main():
    from nkl import agents, fsim
    folder, seed, width, height = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    video = (int(sys.argv[5]), int(sys.argv[6]))
    run = context = pacing = None
    motor = None
    while line := commands.readline():
        command = json.loads(line)
        try:
            if 'reset' in command:
                settings = command['reset']
                pacing = pacing_of(*settings['pacing'])
                limits = agents.Limits(**{key: value for key, value in settings['limits'].items() if value is not None})
                if settings['limits'].get('actions') is None and isinstance(pacing, agents.Dense):
                    limits = replace(limits, actions=1)       # nkl's own rule: one action a turn under Dense
                motor = agents.Motor(**(settings.get('motor') or {}))
                run, context = agents.begin(start(folder, seed, (width, height), settings.get('picture'),
                                                  settings.get('shown'), settings.get('main', ())), limits=limits)
                system = agents.system_prompt(
                    settings['instructions'], screen=(context.screen.width, context.screen.height),
                    extent=context.extent, limits=limits, pacing=pacing, motor=motor)
                send({'done': screen(run, context, system)})
            elif 'step' in command:
                step = command['step']
                made = actions(step['acts'], context)
                response = agents.AgentResponse(tuple(made), (), (), '', None, agents.Usage(), float(step['seconds']),
                                                1, '', '', '')
                frames, run_schedule = Frames(step['frames']) if step.get('frames') else None, fsim.run
                if frames is not None:
                    fsim.run = with_frames(video, frames)
                try:
                    run, images = agents.act(run, context, response, pacing, motor)
                finally:
                    fsim.run = run_schedule
                context = agents.observe(context, response, images)
                send({'done': screen(run, context, frames=frames)})
            elif 'leave' in command:
                if run.outcome is None:
                    run = fsim.stop(run)
                send({'done': True})
            elif 'trace' in command:
                Path(command['trace']).write_text(json.dumps(fsim.trace(run)))
                send({'done': command['trace']})
            else:
                send({'done': run.outcome.get('value') if run.outcome is not None
                      and run.outcome.get('kind') == 'completed' else None})
        except Exception as error:
            send({'failed': {'type': type(error).__name__, 'message': str(error),
                             'traceback': traceback.format_exc()}})


if __name__ == '__main__':
    main()
