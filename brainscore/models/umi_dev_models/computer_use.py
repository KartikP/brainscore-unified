"""Qwen3-VL's own trained computer-use interface for a task on a screen (owner, 2026-11-25: a model family trained
on its own action format may be given the task in that format; its plugin turns its actions into motor acts).

Source (public): the Qwen3-VL authors' cookbook, QwenLM/Qwen3-VL ``cookbooks/utils/agent_function_call.py`` (the
``computer_use`` tool) and ``cookbooks/computer_use.ipynb`` (the system turn by Qwen-Agent's NousFnCallPrompt,
the screen stated as 1000 x 1000, coordinates on a 0-1000 grid of the picture). The model gets the tool in its
system turn in the authors' words, and each turn one task line (``TASK_LINE`` and the harness's
``observation['task']``: the act form and the benchmark's instructions; nkl's own briefing is not used), the
screen (and at most ``FRAMES_SHOWN`` changed frames since its last turn) and its last turns as text. It answers with ``<tool_call>`` blocks; each becomes motor acts:

- ``mouse_move``: move; ``left_click``: move to its coordinate (if given), down, up (``double_click`` and
  ``triple_click``: two and three clicks there); ``left_click_drag``: down at
  the pointer, a stroke to the coordinate, up (the drag starts where the pointer is: the cookbook states only the
  end); ``key``: press the keys in order, release them in reverse; ``type``: type; ``wait``: wait (seconds).
- ``terminate``: no act (the run goes on until the task ends).
- several calls in one reply are taken in order, up to ``MAX_ACTS`` acts.
- ``right_click``, ``middle_click``, ``scroll``, ``hscroll``, ``answer``, more than ``MAX_ACTS`` acts,
  an action of the other act form, a key the task cannot take: the reply is invalid (None), with the action's
  name; it is shown to the model at its next turn.

Limit: key names are read by ``KEYS`` (the common names: enter, space, arrows, tab, escape, letters, digits);
another key name makes the reply invalid with the name.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

import numpy as np

COMMIT = '0dcc180d854f4b132f8059b10bbf0b5fd5dae9ed'      # the last commit of the authors' agent_function_call.py
FORMAT = f'computer_use (QwenLM/Qwen3-VL {COMMIT[:8]}, cookbooks/utils/agent_function_call.py, Apache-2.0)'
GRID = 1000                     # the screen as the format states it: 1000 x 1000
TURNS_KEPT = 8
FRAMES_SHOWN = 4
MAX_ACTS = 20                   # the participant view's acts of one step
TASK_LINE = 'Do the task on the screen.'

# The authors' words, as their cookbook gives them (QwenLM/Qwen3-VL at COMMIT, Apache License 2.0): the
# ``computer_use`` tool's description (display 1000 x 1000) and parameters, and Qwen-Agent's FN_CALL_TEMPLATE
# (QwenLM/Qwen-Agent qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py, Apache License 2.0) for the system turn.
_DESCRIPTION = """Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is 1000x1000.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges."""
_ACTIONS = """The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question."""
TOOL = {'type': 'function', 'function': {'name': 'computer_use', 'description': _DESCRIPTION, 'parameters': {
    'properties': {
        'action': {'description': _ACTIONS,
                   'enum': ['key', 'type', 'mouse_move', 'left_click', 'left_click_drag', 'right_click',
                            'middle_click', 'double_click', 'triple_click', 'scroll', 'hscroll', 'wait',
                            'terminate', 'answer'],
                   'type': 'string'},
        'keys': {'description': 'Required only by `action=key`.', 'type': 'array'},
        'text': {'description': 'Required only by `action=type` and `action=answer`.', 'type': 'string'},
        'coordinate': {'description': '(x, y): The x (pixels from the left edge) and y (pixels from the top edge) '
                                      'coordinates to move the mouse to.', 'type': 'array'},
        'pixels': {'description': 'The amount of scrolling to perform. Positive values scroll up, negative values '
                                  'scroll down. Required only by `action=scroll` and `action=hscroll`.',
                   'type': 'number'},
        'time': {'description': 'The seconds to wait. Required only by `action=wait`.', 'type': 'number'},
        'status': {'description': 'The status of the task. Required only by `action=terminate`.', 'type': 'string',
                   'enum': ['success', 'failure']}},
    'required': ['action'], 'type': 'object'}}}
FN_CALL_TEMPLATE = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
# The cookbook's system turn: its system message, then the tools (NousFnCallPrompt appends them after '\n\n').
SYSTEM = 'You are a helpful assistant.' + '\n\n' + FN_CALL_TEMPLATE.format(
    tool_descs=json.dumps(TOOL, ensure_ascii=False))

# The key names the format's examples use, as W3C KeyboardEvent.code (a name outside the map is an invalid answer).
KEYS = {'enter': 'Enter', 'return': 'Enter', 'space': 'Space', ' ': 'Space', 'left': 'ArrowLeft',
        'right': 'ArrowRight', 'up': 'ArrowUp', 'down': 'ArrowDown', 'arrowleft': 'ArrowLeft',
        'arrowright': 'ArrowRight', 'arrowup': 'ArrowUp', 'arrowdown': 'ArrowDown', 'tab': 'Tab',
        'escape': 'Escape', 'esc': 'Escape', 'backspace': 'Backspace',
        **{c: f'Key{c.upper()}' for c in 'abcdefghijklmnopqrstuvwxyz'}, **{d: f'Digit{d}' for d in '0123456789'}}
POINTER, KEY = {'mouse_move', 'left_click', 'left_click_drag', 'double_click', 'triple_click'}, {'key', 'type'}
CALL = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)


def read(reply: str, form: str, pointer, size) -> tuple:
    """The reply's ``computer_use`` calls as motor acts of ``form`` (positions: pixels of a ``size`` picture), and
    the problems (None acts when there is one). ``pointer`` is where the pointer is, in pixels."""
    width, height = size
    acts, problems, at = [], [], list(pointer)
    calls = CALL.findall(reply)
    if not calls:
        return None, ['the reply holds no <tool_call> of computer_use']
    for number, raw in enumerate(calls, 1):
        try:
            call = json.loads(raw)
            name, arguments = call['name'], call['arguments']
            action = arguments['action']
        except (ValueError, KeyError, TypeError):
            return None, [f'call {number} is not {{"name": "computer_use", "arguments": {{"action": ...}}}}: {raw[:120]}']
        if name != 'computer_use':
            return None, [f'call {number} names {name!r}, not computer_use']
        wanted = POINTER if form == 'motor:pointer' else KEY
        if action not in wanted | {'wait', 'terminate'}:
            return None, [f'call {number}: {action!r} cannot be done in this task (it takes '
                          f'{"pointer" if form == "motor:pointer" else "keyboard"} actions)']
        coordinate = arguments.get('coordinate')
        if coordinate is not None:
            try:
                x, y = (float(v) for v in coordinate)
            except (TypeError, ValueError):
                return None, [f'call {number}: coordinate {coordinate!r} is not (x, y)']
            point = [x * width / GRID, y * height / GRID]
        elif action in ('mouse_move', 'left_click_drag'):
            return None, [f'call {number}: {action} needs a coordinate']
        if action == 'mouse_move':
            acts.append({'act': 'move', 'x': point[0], 'y': point[1]}); at = point
        elif action in ('left_click', 'double_click', 'triple_click'):
            if coordinate is not None:
                acts.append({'act': 'move', 'x': point[0], 'y': point[1]}); at = point
            acts += [{'act': 'down'}, {'act': 'up'}] * {'left_click': 1, 'double_click': 2, 'triple_click': 3}[action]
        elif action == 'left_click_drag':
            acts += [{'act': 'down'}, {'act': 'stroke', 'points': [point]}, {'act': 'up'}]; at = point
        elif action == 'key':
            keys = arguments.get('keys')
            keys = [keys] if isinstance(keys, str) else keys
            codes = [KEYS.get(str(key).lower()) for key in keys or []]
            if not codes or None in codes:
                return None, [f'call {number}: key {keys!r} is not a key name this task reads']
            acts += [{'act': 'press', 'key': code} for code in codes]
            acts += [{'act': 'release', 'key': code} for code in reversed(codes)]
        elif action == 'type':
            acts.append({'act': 'type', 'text': str(arguments.get('text', ''))})
        elif action == 'wait':
            acts.append({'act': 'wait', 'ms': float(arguments.get('time', 1)) * 1000})
    if len(acts) > MAX_ACTS:
        return None, [f'the reply makes {len(acts)} acts; at most {MAX_ACTS} a turn']
    return acts, problems


def computer_use_policy(reply: Callable, *, max_tokens: int = 1024, temperature: Optional[float] = None,
                        seed: Optional[int] = None) -> Callable:
    """A ``policy(row) -> {act form: acts}`` that asks ``reply`` in Qwen3-VL's computer-use format."""
    import tempfile
    from pathlib import Path
    from PIL import Image
    from brainscore_core.responses import Prompt, Text
    from brainscore_core.streaming import StreamEvent
    from brainscore.harnesses.nodekit.policy import changed_frames
    from brainscore.model_helpers.provenance import state_settings_as
    state = {'history': [], 'last': None, 'problems': [], 'turn': 0, 'folder': None}
    trace: list = []

    def picture(array) -> str:
        if state['folder'] is None:
            state['folder'] = Path(tempfile.mkdtemp(prefix='computer-use-'))
        path = state['folder'] / f'{len(list(state["folder"].iterdir())):06d}.png'
        Image.fromarray(np.asarray(array)).save(path)
        return str(path)

    def policy(row):
        seen, screen = row['observation'], row['vision:image']
        video = row.get('vision:video')
        shown = changed_frames(video, seen.get('frame_t_ms', []), state['last'], FRAMES_SHOWN)
        parts, names = [('text', SYSTEM)], ['instruction']
        text = [f"{TASK_LINE} {seen.get('task', '')}".strip()]
        if state['problems']:
            text.append('Your last reply was not done: ' + '; '.join(state['problems']))
        parts.append(('text', '\n'.join(text))); names.append('task')
        for at, frame in shown:
            parts += [('text', f'A frame since your last turn, at {at / 1000:.3f} s:'), ('image', picture(frame))]
            names += ['frame', 'frame']
        parts += [('text', 'The screen now:'), ('image', picture(screen))]
        names += ['turn', 'screen']
        settings = Text(max_tokens=max_tokens, temperature=temperature,
                        seed=None if seed is None else seed + state['turn'])
        answer = reply(Prompt(parts=tuple(parts), names=tuple(names), response=settings,
                              history=tuple(state['history'][-TURNS_KEPT:])))
        state_settings_as(policy, reply, {'action_format': FORMAT, 'position_grid': GRID})
        said = answer.payload if isinstance(answer, StreamEvent) else answer
        acts, problems = read(said, seen['act_form'], seen.get('pointer', (0, 0)), (screen.shape[1], screen.shape[0]))
        state['history'].append((Prompt(parts=(parts[1], ('text', 'The screen now (not shown again).')),
                                        names=('task', 'turn')), said))
        state.update(last=video[-1] if video is not None and len(video) else state['last'], problems=problems,
                     turn=state['turn'] + 1)
        trace.append({'turn': state['turn'], 'reply': said, 'acts': acts, 'problems': problems})
        return {seen['act_form']: StreamEvent(seen['act_form'], acts, None, {
            'answer': said, 'valid': not problems,
            'parse_error': '; '.join(problems) or None})}

    def reset():
        state.update(history=[], last=None, problems=[], turn=0)
        trace.clear()

    policy.reset, policy.trace, policy.close = reset, trace, lambda: None
    return policy
