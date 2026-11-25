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
it; the first is compared with the last screen), spread evenly over the changed ones, each with its time.
The reply is read with nkl's ``parse`` in the same child; its acts go out on the act form the run reads
(``observation['act_form']``) in the numbers the reply wrote. A reply with a problem (nkl's rules, a look,
an act of the other form) gives no act (None, an invalid step) and its problems with the command and the
rule broken are shown at the next turn, as nkl does; the event meta holds the reply and the problems.

The model sees no earlier turn: what it keeps is nkl's memory notes (at most 16 of 400 characters, nkl's
``Limits``), kept here and cleared in ``reset()``. ``trace`` holds each turn's reply, problems, acts and
token use.

Limit: the policy needs NodeKit's task repository (private) for nkl's environment, as the harness does.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

FRAMES_SHOWN = 4          # the most frames of vision:video one turn shows the model
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


def act_policy(reply: Callable, *, frames: int = FRAMES_SHOWN, max_tokens: int = 4096,
               temperature: Optional[float] = None, seed: Optional[int] = None) -> Callable:
    """A ``policy(row) -> {act form: acts}`` that asks ``reply`` (a ``reply(prompt)`` part) each turn and
    reads its answer with nkl's reader. ``max_tokens``, ``temperature`` and ``seed`` are the ``Text`` response
    settings of each turn (None: the model's default, greedy for a local model); with a seed, turn ``n`` uses
    ``seed + n``, so a repeated run with the same seed asks the same."""
    from brainscore_core.responses import Prompt, Text
    from brainscore_core.streaming import StreamEvent
    reader, folder = _Reader(), Path(tempfile.mkdtemp(prefix='nodekit-turns-'))
    state = {'memories': [], 'problems': [], 'last': None, 'turn': 0}
    trace: list = []

    def picture(array) -> str:
        from PIL import Image
        path = folder / f'{state["turn"]:05d}-{len(list(folder.iterdir()))}.png'
        Image.fromarray(np.asarray(array)).save(path)
        return str(path)

    def policy(row):
        seen, screen = row['observation'], row['vision:image']
        turn = {'screen': [screen.shape[1], screen.shape[0]], 'observation': seen,
                'memories': state['memories'], 'problems': state['problems']}
        parts, names = [('text', row['instruction'])], ['instruction']
        shown = changed_frames(row.get('vision:video'), seen.get('frame_t_ms', []), state['last'], frames)
        briefing = reader.ask({'brief': turn})
        for number, (text, image) in enumerate(briefing):
            if image and shown:          # the screen comes last: the frames since the last turn before it
                for at, frame in shown:
                    parts += [('text', f'A frame since your last turn, at {at / 1000:.3f} s:'),
                              ('image', picture(frame))]
                    names += ['frame', 'frame']
            parts.append(('text', text))
            names.append('turn')
            if image:
                parts.append(('image', picture(screen)))
                names.append('screen')
        settings = Text(max_tokens=max_tokens, temperature=temperature,
                        seed=None if seed is None else seed + state['turn'])
        answer = reply(Prompt(parts=tuple(parts), names=tuple(names), response=settings))
        text = answer.payload if isinstance(answer, StreamEvent) else answer
        meta = answer.meta if isinstance(answer, StreamEvent) else {}
        read = reader.ask({'read': {**turn, 'text': text, 'form': seen['act_form']}})
        state.update(memories=read['memories'], problems=read['problems'], last=screen, turn=state['turn'] + 1)
        trace.append({'turn': state['turn'], 't_ms': seen['t_ms'], 'frames_shown': [at for at, _ in shown],
                      'reply': text, 'problems': read['problems'], 'acts': read['acts'],
                      'token_usage': meta.get('token_usage'), 'reasoning': meta.get('reasoning')})
        problems = '; '.join(read['problems']) or None
        return {seen['act_form']: StreamEvent(seen['act_form'], read['acts'], None, {
            'answer': text, 'valid': problems is None,
            'parse_error': None if problems is None else f'the reply broke nkl\'s rules: {problems}'})}

    def reset():
        state.update(memories=[], problems=[], last=None, turn=0)
        trace.clear()

    policy.reset, policy.trace, policy.close = reset, trace, reader.close
    return policy
