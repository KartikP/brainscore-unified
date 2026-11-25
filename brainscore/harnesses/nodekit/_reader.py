"""nkl's reply reader and turn briefing for a text-answering subject, in nkl's own environment.

    <environment python> _reader.py

Standard library and nkl only; no run, no server. Commands, one JSON line each, with the facts of the turn
(``turn``: the screen size, the harness's ``observation`` row entry, the memories and the problems of the
last reply):

- ``{"brief": turn}`` answers nkl's briefing of the turn as parts ``[text, picture]`` (the picture's name
  where one goes: ``screen`` or ``look:<id>``).
- ``{"system": {"briefing": ..., "screen": [w, h], "extent": [w, h]}}`` answers nkl's system briefing written
  for that screen (a subject's own grid).
- ``{"read": {"text": reply, "form": "motor:key" | "motor:pointer", **turn}}`` reads the reply with nkl's
  ``parse`` and answers ``{"acts": [...] or None, "memories": [...], "problems": [...]}``: the acts in the
  form of Brain-Score's ``motor:key`` or ``motor:pointer`` (the numbers the reply wrote, in the units of the
  turn's screen), or None
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
    context = context_of(turn)
    text, form = turn['text'], turn['form']
    actions, changes, problems = agents.parse(text, context)
    problems = list(problems)
    # The numbers the reply wrote for each pointer move, in order: a stroke's points are these pixels.
    pixels = iter([[command['x'], command['y']] for _, command in written(text)
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


def system(command):
    """nkl's system briefing for the harness's arguments (``observation['briefing']``) on a screen of
    ``command['screen']`` (a subject's grid): nkl writes its own sentences for that screen; the stroke times
    are for ``command['extent']`` scene units across the screen's width."""
    from dataclasses import replace
    from nkl import agents
    settings = command['briefing']
    kind, step = settings['pacing']
    pacing = {'Frozen': agents.Frozen(), 'Realtime': agents.Realtime(), 'Dense': agents.Dense(step)}[kind]
    limits = agents.Limits(**{key: value for key, value in settings['limits'].items() if value is not None})
    if settings['limits'].get('actions') is None and isinstance(pacing, agents.Dense):
        limits = replace(limits, actions=1)
    return agents.system_prompt(settings['instructions'], screen=tuple(command['screen']),
                                extent=tuple(command['extent']), limits=limits, pacing=pacing,
                                motor=agents.Motor(), seconds=settings['seconds'])


def main():
    while line := commands.readline():
        command = json.loads(line)
        try:
            send({'done': brief(command['brief']) if 'brief' in command else
                  system(command['system']) if 'system' in command else read(command['read'])})
        except Exception as error:
            send({'failed': {'type': type(error).__name__, 'message': str(error),
                             'traceback': traceback.format_exc()}})


if __name__ == '__main__':
    main()
