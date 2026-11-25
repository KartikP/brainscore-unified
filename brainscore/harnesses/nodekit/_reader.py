"""nkl's reply reader and turn briefing for a text-answering subject, in nkl's own environment.

    <environment python> _reader.py

Standard library and nkl only; no run, no server. Commands, one JSON line each, with the facts of the turn
(``turn``: the screen size, the harness's ``observation`` row entry, the memories and the problems of the
last reply):

- ``{"brief": turn}`` answers nkl's briefing of the turn as parts ``[text, picture]`` (the picture's name
  where one goes: ``screen`` or ``look:<id>``).
- ``{"read": {"text": reply, "form": "motor:key" | "motor:pointer", **turn}}`` reads the reply with nkl's
  ``parse`` and answers ``{"acts": [...] or None, "memories": [...], "problems": [...]}``: the acts in the
  form of Brain-Score's ``motor:key`` or ``motor:pointer`` (pixels: the numbers the reply wrote; with ``grid``,
  the subject's grid of that many positions a side, scaled to the screen), or None
  and nkl's problems with the command and the rule broken. In the agent view a look is an act
  (``{'act': 'look', 'id'}``); in the participant view a reply that asks for one has a problem, as has one
  with an act of the other form; as in nkl, nothing of such a reply is done.

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
    looks = tuple((name, fsim.Image(width, height, 'rgb', round(at * 1000), 0, b''))
                  for name, at in seen.get('looks', {}).items())
    return agents.AgentContext(screen, extent, tuple(map(tuple, turn['memories'])), looks, body,
                               tuple(turn['problems']), limits, seen['turns_left'])


def brief(turn):
    """nkl's briefing as parts ``[text, picture name]``: ``screen``, ``look:<id>`` or None."""
    from nkl._core.host.agents import briefing
    context = context_of(turn)
    names = {id(image): f'look:{name}' for name, image in context.looks}
    names[id(context.screen)] = 'screen'
    return [[text, None if image is None else names[id(image)]] for text, image in briefing(context)]


KEY_ACTS = {'Press': 'press', 'Release': 'release'}
POINTER_ACTS = {'Down': 'down', 'Up': 'up'}


def read(turn):
    from nkl import agents, fsim
    from nkl._core.host.agents import Stroke, TakeLook, remember, written
    # A subject that writes positions on a grid of ``grid`` per side (its own convention) is read on that grid,
    # and its positions become pixels of the screen.
    grid = turn.get('grid')
    width, height = turn['screen']
    context = context_of(dict(turn, screen=[grid, grid]) if grid else turn)
    text, form = turn['text'], turn['form']
    actions, changes, problems = agents.parse(text, context)
    problems = list(problems)
    # The numbers the reply wrote for each pointer move, in order: a stroke's points are these pixels.
    scale = (width / grid, height / grid) if grid else (1, 1)
    pixels = iter([[command['x'] * scale[0], command['y'] * scale[1]] for _, command in written(text)
                   if isinstance(command, dict) and command.get('do') == 'pointer_move'])
    acts = []
    for action in actions:
        if isinstance(action, TakeLook) and turn['observation'].get('view') == 'agent':
            acts.append({'act': 'look', 'id': action.id})
        elif isinstance(action, TakeLook):
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
        other = next(kind for kind, names in kinds.items() if act['act'] in names) if act['act'] not in (
            'wait', 'look') else wanted
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
