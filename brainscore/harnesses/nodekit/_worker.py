"""One nkl program (a bundle's ``program.json``) as a simulated participant, in nkl's own environment.

    <environment python> _worker.py <bundle folder> <seed> <screen width> <screen height>

Standard library and nkl only. A step is one turn of nkl's agent harness (``nkl.agents``): the turn's
acts as one schedule (``act``), then a look at the screen, under the run's pacing; the body (pointer, keys)
follows the acts (``observe``). Commands, one JSON line each:

- ``{"reset": {"pacing": [kind, step µs], "limits": {...}, "seconds": s, "instructions": text}}`` starts
  the run (``begin``) and answers the first screen with nkl's briefing for it (``system_prompt``).
- ``{"step": {"acts": [act, ...], "seconds": s}}``: the acts in the form of Brain-Score's ``motor:key`` and
  ``motor:pointer`` (pixels of the screen, origin top left), ``seconds`` the subject's answering time (it
  passes in the run only under the Realtime pacing).
- ``{"leave": true}`` stops a run that is still going, as nkl's agent harness does at its end.
- ``{"result": true}``: the program's result as value JSON, or None when it did not complete.
- ``{"trace": <file>}`` writes the run's trace lines to the file as one JSON array.

A reset or a step answers ``{"done": screen}``: the picture (base64 RGB, top row first), the run's time
in ms, whether it has ended, and the body and limits the subject's briefing shows.

Limit: nkl 0.1.0 starts a run from a ``Call`` of Python code; a bundle holds the lowered program
instead. ``BundleCall`` gives the program, its arguments and its media as a ``Call`` does, and the
worker lets ``nkl.fsim`` take it (``op_call``) and finds the bundle's media files (``nkl.media._files``).
It converts pixels with nkl's own ``to_scene`` and ``to_pixels`` and makes strokes with its ``Stroke``. These are names inside nkl 0.1.0;
another nkl version must be checked again.
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


def start(folder: Path, seed: int, screen):
    """A fresh run of the bundle, as ``nkl.fsim.start`` makes one for nkl's agent harness."""
    import nkl._core.frontend.program as program
    from nkl import fsim
    call = BundleCall(folder)
    op_call = program.op_call
    program.op_call = lambda value, use: value if value is call else op_call(value, use)
    try:
        return fsim.start(call, seed=seed, display=fsim.Display(screen=screen), look=fsim.Look(channels='rgb'))
    finally:
        program.op_call = op_call


def pacing_of(kind, step):
    from nkl import agents
    return {'Frozen': agents.Frozen(), 'Realtime': agents.Realtime(), 'Dense': agents.Dense(step)}[kind]


def actions(acts, context):
    """nkl's agent actions for Brain-Score acts: pixels become scene units by nkl's ``to_scene``; a typed
    text becomes its key presses as nkl's reply reader makes them."""
    from nkl import agents, fsim
    from nkl._core.host.agents import Stroke, to_scene
    made = []

    def scene(x, y):
        return to_scene(x, y, context.screen, context.extent)
    for act in acts:
        name = act['act']
        if name == 'press':
            made.append(fsim.Keyboard.Press(act['key']))
        elif name == 'release':
            made.append(fsim.Keyboard.Release(act['key']))
        elif name == 'type':
            typed, _, problems = agents.parse(json.dumps({'do': 'type', 'text': act['text']}), context)
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
        elif name == 'stroke':
            made.append(Stroke(tuple(scene(*point) for point in act['points']), act.get('force', 1.0)))
        else:
            raise ValueError(f'{name!r} is not an act of motor:key or motor:pointer')
    return made


def screen(run, context, system=None):
    from nkl._core.host.agents import to_pixels
    ended = run.outcome is not None
    image, body = context.screen, context.body
    x, y = to_pixels(body.pointer.x, body.pointer.y, image, context.extent)
    answer = {'t_ms': run.t / 1000, 'ended': ended,
              'body': {'pointer': [x, y], 'pointer_down': body.pointer.down, 'pointer_connected': body.pointer.connected,
                       'keys_down': list(body.keyboard.down)},
              'extent': list(context.extent), 'acts_per_step': context.limits.actions}
    if not ended:
        answer['picture'] = {'data': base64.b64encode(image.data).decode(), 'shape': [image.height, image.width, 3],
                             'at_ms': image.at / 1000}
    if system is not None:
        answer['system'] = system
    return answer


def main():
    from nkl import agents, fsim
    folder, seed, width, height = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    run = context = pacing = None
    motor = agents.Motor()
    while line := commands.readline():
        command = json.loads(line)
        try:
            if 'reset' in command:
                settings = command['reset']
                pacing = pacing_of(*settings['pacing'])
                limits = agents.Limits(**settings['limits'])
                if isinstance(pacing, agents.Dense):
                    limits = replace(limits, actions=1)       # as nkl's agent harness: one action a turn
                run, context = agents.begin(start(folder, seed, (width, height)), limits=limits)
                system = agents.system_prompt(
                    settings['instructions'], screen=(context.screen.width, context.screen.height),
                    extent=context.extent, limits=limits, pacing=pacing, motor=motor, seconds=settings['seconds'])
                send({'done': screen(run, context, system)})
            elif 'step' in command:
                step = command['step']
                made = actions(step['acts'], context)
                response = agents.AgentResponse(tuple(made), (), (), '', None, agents.Usage(), float(step['seconds']),
                                                1, '', '', '')
                run, images = agents.act(run, context, response, pacing, motor)
                context = agents.observe(context, response, images)
                send({'done': screen(run, context)})
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
