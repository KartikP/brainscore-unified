"""nkl's reply reader and turn briefing for a text-answering subject, in nkl's own environment.

    <environment python> _reader.py

Standard library and nkl only; no run, no server. Commands, one JSON line each, with the facts of the turn
(``turn``: the screen size, the harness's ``observation`` row entry, the memories and the problems of the
last reply):

- ``{"brief": turn}`` answers nkl's briefing of the turn as parts ``[text, image]`` (``image`` true where the
  screen goes).
- ``{"read": {"text": reply, "form": "motor:key" | "motor:pointer", **turn}}`` reads the reply with nkl's
  ``parse`` and answers ``{"acts": [...] or None, "memories": [...], "problems": [...]}``: the acts in the
  form of Brain-Score's ``motor:key`` or ``motor:pointer`` (pixels: the numbers the reply wrote), or None
  and nkl's problems with the command and the rule broken. A reply that asks for a look (not offered) or
  for an act of the other form has a problem; as in nkl, nothing of such a reply is done.

Limit: the briefing and the reader are nkl 0.1.0's ``briefing``, ``parse``, ``written`` and ``remember`` (names
inside ``nkl._core.host.agents``); another nkl version must be checked again.
"""
import json
import os
import traceback

read_fd, write_fd = (int(fd) for fd in os.environ['BRAINSCORE_CHILD_FDS'].split(','))
commands, messages = os.fdopen(read_fd, 'rb'), os.fdopen(write_fd, 'wb')


def send(message):
    messages.write((json.dumps(message) + '\n').encode())
    messages.flush()


def context_of(turn):
    """nkl's agent context for the turn: one scene unit a pixel (the harness refuses another extent)."""
    from nkl import agents, fsim
    from nkl._core.host.agents import to_scene
    width, height = turn['screen']
    seen = turn['observation']
    screen = fsim.Image(width, height, 'rgb', round(seen['t_ms'] * 1000), 0, b'')
    extent = (float(width), float(height))
    x, y = to_scene(*seen['pointer'], screen, extent)
    body = agents.Body(fsim.Pointer.State(x, y, seen['pointer_down'], seen['pointer_connected']),
                       fsim.Keyboard.State(tuple(seen['keys_down']), True))
    limits = agents.Limits(actions=seen['acts_per_step'])
    return agents.AgentContext(screen, extent, tuple(map(tuple, turn['memories'])), (), body,
                               tuple(turn['problems']), limits, seen['turns_left'])


def brief(turn):
    from nkl._core.host.agents import briefing
    return [[text, image is not None] for text, image in briefing(context_of(turn))]


KEY_ACTS = {'Press': 'press', 'Release': 'release'}
POINTER_ACTS = {'Down': 'down', 'Up': 'up'}


def read(turn):
    from nkl import agents, fsim
    from nkl._core.host.agents import Stroke, TakeLook, remember, written
    context = context_of(turn)
    text, form = turn['text'], turn['form']
    actions, changes, problems = agents.parse(text, context)
    problems = list(problems)
    # The numbers the reply wrote for each pointer move, in order: a stroke's points are these pixels.
    pixels = iter([[command['x'], command['y']] for _, command in written(text)
                   if isinstance(command, dict) and command.get('do') == 'pointer_move'])
    acts = []
    for action in actions:
        if isinstance(action, TakeLook):
            problems.append(f'look ({action.id}) is not offered here: you see the screen when the reply has played out')
        elif isinstance(action, agents.Wait):
            acts.append({'act': 'wait', 'ms': action.us / 1000})
        elif isinstance(action, Stroke):
            acts.append({'act': 'stroke', 'points': [next(pixels) for _ in action.points],
                         **({} if action.force == 1.0 else {'force': action.force})})
        elif isinstance(action, (fsim.Keyboard.Press, fsim.Keyboard.Release)):
            acts.append({'act': KEY_ACTS[type(action).__name__], 'key': action.key})
        else:
            acts.append({'act': POINTER_ACTS[type(action).__name__]})
    kinds = {'key': {'press', 'release'}, 'pointer': {'stroke', 'down', 'up'}}
    wanted = form.split(':')[1]
    for act in acts:
        other = next(kind for kind, names in kinds.items() if act['act'] in names) if act['act'] != 'wait' else wanted
        if other != wanted:
            problems.append(f'{act["act"]} is a {other} action; this task takes {wanted} actions only')
            break
    if problems:
        return {'acts': None, 'memories': turn['memories'], 'problems': problems}
    return {'acts': acts, 'memories': [list(m) for m in remember(context.memories, changes, context.limits)],
            'problems': []}


def main():
    while line := commands.readline():
        command = json.loads(line)
        try:
            send({'done': brief(command['brief']) if 'brief' in command else read(command['read'])})
        except Exception as error:
            send({'failed': {'type': type(error).__name__, 'message': str(error),
                             'traceback': traceback.format_exc()}})


if __name__ == '__main__':
    main()
