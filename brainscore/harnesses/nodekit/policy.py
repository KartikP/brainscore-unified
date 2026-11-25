"""One policy for every text-answering subject on a NodeKit task: a local vision-language model or a hosted
model API, through its ``reply(prompt)`` part.

    reply = build_api_reply('bedrock', 'us.anthropic.claude-sonnet-5-5')   # or chat_parts(model, processor)[0]
    subject = BrainScoreModel('my-model', policy=act_policy(reply),
                              policy_in_channels=('vision:image', 'vision:video', 'instruction', 'observation'),
                              policy_out_channels=('motor:key', 'motor:pointer'))

Each turn the model gets nkl's own briefing (Decision 2: NodeKit's words, read from nkl at run time, nothing
of them in this file): the harness's ``instruction`` is the system text (nkl's ``system_prompt``), and the
turn is nkl's ``briefing`` (turns left, the problems of the last reply, the memories, the pointer and keys,
the screen now), made in a child process in nkl's environment (``_reader.py``). Before the screen, the turn
shows at most ``frames`` frames of ``vision:video`` that changed (a frame that differs from the one before
it; the first is compared with the last frame of the turn before), spread evenly over the changed ones, each
with its time; the trace states the times shown.
The reply is read with nkl's ``parse`` in the same child; its acts go out on the act form the run reads
(``observation['act_form']``) in the numbers the reply wrote. A reply with a problem (nkl's rules, a look,
an act of the other form) gives no act (None, an invalid step) and its problems with the command and the
rule broken are shown at the next turn, as nkl does; the event meta holds the reply and the problems.

In the participant view the conversation is kept within a stated limit (in the agent view, nkl's own, the model
sees no earlier turn): the last ``turns_kept`` turns come before the turn as earlier
user and assistant turns, as text only (their briefing and the reply; their pictures are not sent again). The
model also keeps nkl's memory notes (at most 16 of 400 characters, nkl's ``Limits``). Both are cleared in
``reset()``. ``trace`` holds each turn's reply, problems, acts and
token use.

Limit: the policy needs NodeKit's task repository (private) for nkl's environment, as the harness does.
"""
from __future__ import annotations

import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Callable, Optional

from brainscore.model_helpers.provenance import state_settings_as

import numpy as np

FRAMES_SHOWN = 4          # the most frames of vision:video one turn shows the model
TURNS_KEPT = 8            # the earlier turns, as text, that each turn shows the model
_READER = Path(__file__).with_name('_reader.py')


class _Reader:
    """nkl's briefing and reply reader in a child process (started at the first turn)."""

    def __init__(self):
        self._child = None

    def ask(self, command: dict):
        if self._child is None:
            import os
            from brainscore.harnesses._child_process import ChildProcess
            from brainscore.harnesses._environment_folder import _RUNTIME_VARIABLES, environment_python, ready_folder
            from brainscore.harnesses.nodekit import library
            folder = ready_folder(library(), 'NodeKit reader')
            environment = {name: os.environ[name] for name in _RUNTIME_VARIABLES if name in os.environ}
            environment.update(PYTHONNOUSERSITE='1')
            self._child = ChildProcess([environment_python(folder), _READER], env=environment,
                                       log_directory=folder / 'logs', idle_timeout_s=120.0).__enter__()
        return self._child.ask(command)

    def close(self):
        if self._child is not None:
            self._child.__exit__(None, None, None)
            self._child = None


def on_grid(turn: dict, grid: int) -> dict:
    """The turn as the model's grid shows it: a screen of ``grid`` x ``grid`` and the pointer in grid units."""
    width, height = turn['screen']
    seen = dict(turn['observation'])
    if 'pointer' in seen:
        seen['pointer'] = [seen['pointer'][0] * grid / width, seen['pointer'][1] * grid / height]
    return dict(turn, screen=[grid, grid], observation=seen)


def to_pixels(acts: list, grid: int, width: int, height: int) -> list:
    """The acts' positions on the model's grid as pixels of a ``width`` x ``height`` picture."""
    def pixel(x, y):
        return x * width / grid, y * height / grid
    converted = []
    for act in acts:
        act = dict(act)
        if 'x' in act:
            act['x'], act['y'] = pixel(act['x'], act['y'])
        if 'points' in act:
            act['points'] = [list(pixel(*point)) for point in act['points']]
        converted.append(act)
    return converted


def changed_frames(video, times, last, most: int):
    """The frames of ``video`` that differ from the frame before them (the first from ``last``), at most
    ``most`` of them, spread evenly: ``[(time ms, frame)]``."""
    if video is None:
        return []
    changed, before = [], last
    for frame, at in zip(video, times):
        if before is None or frame.shape != before.shape or not np.array_equal(frame, before):
            changed.append((at, frame))
        before = frame
    if len(changed) <= most:
        return changed
    return [changed[round(i * (len(changed) - 1) / (most - 1))] for i in range(most)] if most > 1 else changed[-1:]


def act_policy(reply: Callable, *, frames: int = FRAMES_SHOWN, turns_kept: int = TURNS_KEPT, max_tokens: int = 4096,
               temperature: Optional[float] = None, seed: Optional[int] = None,
               position_grid: Optional[int] = None) -> Callable:
    """A ``policy(row) -> {act form: acts}`` that asks ``reply`` (a ``reply(prompt)`` part) each turn and
    reads its answer with nkl's reader. ``max_tokens``, ``temperature`` and ``seed`` are the ``Text`` response
    settings of each turn (None: the model's default, greedy for a local model); with a seed, turn ``n`` uses
    ``seed + n``, so a repeated run with the same seed asks the same. ``position_grid``: the model writes
    positions on a grid of that many positions a side of the picture (its own convention, stated by its plugin),
    not in pixels; nkl's system briefing is written by nkl for a screen of that grid (its own sentences, from the
    harness's arguments in ``observation['briefing']``) and every position the turn's briefing shows (the
    pointer) is on that grid too, so what the model reads and writes is one convention; the policy scales its
    acts to pixels and states the grid with its settings (the score's provenance). None (the default): pixels.

    Limit: the grid scales x and y differently (1024 -> 1000, 768 -> 1000): nkl's stroke times ('120 ms for 50
    pixels') are stated for distances across the screen's width; a vertical grid unit is shorter (0.768 px
    against 1.024 px)."""
    from brainscore_core.responses import Prompt, Text
    from brainscore_core.streaming import StreamEvent
    reader = _Reader()
    state = {'memories': [], 'problems': [], 'last': None, 'turn': 0, 'history': [], 'folder': None, 'pictures': 0}
    trace: list = []

    def picture(array) -> str:
        """The picture as a file for the reply part, in the episode's folder (made at its first picture)."""
        from PIL import Image
        if state['folder'] is None:
            state['folder'] = Path(tempfile.mkdtemp(prefix='nodekit-turns-'))
        state['pictures'] += 1
        path = state['folder'] / f'{state["pictures"]:06d}.png'
        Image.fromarray(np.asarray(array)).save(path)
        return str(path)

    def clear():
        if state['folder'] is not None:
            shutil.rmtree(state['folder'], ignore_errors=True)
        state.update(folder=None, pictures=0)

    def policy(row):
        seen, pictures = row['observation'], row['vision:image']
        # The agent view names its pictures (the screen and each kept look); the participant view sends the screen.
        pictures = dict(pictures) if isinstance(pictures, dict) else {'screen': pictures}
        screen = pictures['screen']
        turn = {'screen': [screen.shape[1], screen.shape[0]], 'observation': seen,
                'memories': state['memories'], 'problems': state['problems']}
        if position_grid:        # every position the turn shows and reads is on the model's grid
            turn = on_grid(turn, position_grid)
        system = row['instruction']
        if position_grid and 'briefing' in seen:     # nkl's briefing written for the model's grid, by nkl
            if state.get('system_for') != seen['briefing']:
                state['system'] = reader.ask({'system': {'briefing': seen['briefing'], 'screen': [position_grid] * 2,
                                                         'extent': [screen.shape[1], screen.shape[0]]}})
                state['system_for'] = seen['briefing']
            system = state['system']
        parts, names = [('text', system)], ['instruction']
        video = row.get('vision:video')
        shown = changed_frames(video, seen.get('frame_t_ms', []), state['last'], frames)
        briefing = reader.ask({'brief': turn})
        for number, (text, image) in enumerate(briefing):
            if image == 'screen' and shown:     # the frames since the last turn come before the screen
                for at, frame in shown:
                    parts += [('text', f'A frame since your last turn, at {at / 1000:.3f} s:'),
                              ('image', picture(frame))]
                    names += ['frame', 'frame']
            parts.append(('text', text))
            names.append('turn')
            if image:
                parts.append(('image', picture(pictures[image])))
                names.append('screen' if image == 'screen' else 'look')
        settings = Text(max_tokens=max_tokens, temperature=temperature,
                        seed=None if seed is None else seed + state['turn'])
        # The agent view is nkl's own: its agents see no earlier turn (their memories only).
        kept = 0 if seen.get('view') == 'agent' else turns_kept
        history = tuple(state['history'][-kept:]) if kept else ()
        answer = reply(Prompt(parts=tuple(parts), names=tuple(names), response=settings, history=history))
        state_settings_as(policy, reply, {'position_grid': position_grid,
                                          'system_briefing': f'written for a {position_grid} x {position_grid} screen'}
                          if position_grid else None)
        text = answer.payload if isinstance(answer, StreamEvent) else answer
        meta = answer.meta if isinstance(answer, StreamEvent) else {}
        read = reader.ask({'read': {**turn, 'text': text, 'form': seen['act_form']}})
        if position_grid and read['acts']:
            read['acts'] = to_pixels(read['acts'], position_grid, screen.shape[1], screen.shape[0])
        said = [(part, name) for part, name in zip(parts, names) if part[0] == 'text' and name != 'instruction']
        state['history'].append((Prompt(parts=tuple(p for p, _ in said), names=tuple(n for _, n in said)), text))
        state.update(memories=read['memories'], problems=read['problems'], turn=state['turn'] + 1,
                     last=video[-1] if video is not None and len(video) else state['last'])
        trace.append({'turn': state['turn'], 't_ms': seen['t_ms'], 'frames_shown': [at for at, _ in shown],
                      'reply': text, 'problems': read['problems'], 'acts': read['acts'],
                      'token_usage': meta.get('token_usage'), 'reasoning': meta.get('reasoning')})
        problems = '; '.join(read['problems']) or None
        return {seen['act_form']: StreamEvent(seen['act_form'], read['acts'], None, {
            'answer': text, 'valid': problems is None,
            'parse_error': None if problems is None else f'the reply broke nkl\'s rules: {problems}'})}

    def reset():
        clear()
        state.update(memories=[], problems=[], last=None, turn=0, history=[])
        trace.clear()

    def close():
        clear()
        reader.close()

    policy.reset, policy.trace, policy.close = reset, trace, close
    weakref.finalize(policy, close)       # the reader child and the pictures go with the policy
    return policy
