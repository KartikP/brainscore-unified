"""Multi-agent harness: several subjects in one conversation, where each answer
becomes the input of the next subject.

Each subject takes part through ``participate`` in one closed-loop session for
the whole conversation, the same path as every other experiment. The source of
the session is the core closed-loop source (``environment_session``). Its
environment is the subject's seat: the other subjects and the mediator. A
:class:`Mediator` turns each answer into the row of the next subject and holds
the shared state (a conversation, a gridworld, a market book, ...).

    mediator = DialogueMediator(task='agree on a meeting time')
    transcript = multi_agent_rollout([alice, bob], mediator, n_turns=8)

A subject answers on ``behavior:text`` (the words it generates). The mediator
gives that text to the next subject on ``text`` (the words it reads). A subject
is any ``Subject``; a network becomes one as ``BrainScoreModel(identifier,
policy=..., policy_in_channels=('text',), policy_out_channels=('behavior:text',))``. The
caller owns ``reset()`` between conversations, as for ``run_environment``.

One session for each subject for the whole conversation means one ``begin``:
a subject that keeps state (for example a policy with ``reset()``) keeps it
from turn to turn. Session tools work on the run: a ``Witness`` on a subject
sees each of its turns, and an output requested with ``outputs`` in addition
to the answer (for example ``neural:IT``) is recorded during the pass that
made the answer and is returned in the transcript.

``participate`` returns only when its session ends, and every session stays
open for the whole conversation, so each one runs in its own thread. Only one
subject acts at a time: the harness gives the turn, waits for the answer, then
gives the next turn. A run is therefore as deterministic as its subjects.
"""
import contextvars
import queue
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from brainscore_core.contract import Subject
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import environment_session


class Mediator:
    """Routes each answer into the next subject's row and holds shared state.
    Subclass for a specific world.

    ``observe(identifier, incoming)`` builds the row the subject perceives this
    turn (``incoming`` is the previous answer, or ``None`` on the first turn).
    ``apply(identifier, event)`` folds the subject's answer into the shared
    state. A row is a mapping of input channel to payload; leave out a channel
    that has no value this turn.
    """
    task: Optional[str] = None

    def observe(self, identifier: str, incoming: Optional[StreamEvent]) -> Mapping[str, Any]:
        raise NotImplementedError

    def apply(self, identifier: str, event: StreamEvent) -> None:
        raise NotImplementedError


class DialogueMediator(Mediator):
    """A turn-taking conversation.

    Each subject hears the previous answer on ``text`` and the task on
    ``instruction``. The first subject hears ``start``, or no ``text`` when
    ``start`` is None. ``history`` holds every answer in order: a
    ``behavior:text`` event with the speaker in ``meta['sender']``.
    """

    def __init__(self, task: Optional[str] = None, start: Optional[str] = None):
        self.task = task
        self.start = start
        self.history: List[StreamEvent] = []

    def observe(self, identifier: str, incoming: Optional[StreamEvent]) -> Dict[str, Any]:
        heard = self.start if incoming is None else incoming.payload
        row = {} if heard is None else {'text': heard}
        if self.task is not None:
            row['instruction'] = self.task
        return row

    def apply(self, identifier: str, event: StreamEvent) -> None:
        self.history.append(event)


class _Seat:
    """The environment of one subject's session: its turns in the conversation.

    ``reset`` and ``step`` wait until the harness gives the subject its next
    row, or ends the conversation. ``step`` hands the answer to the harness.
    """

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.rows: queue.Queue = queue.Queue()      # harness -> subject: (row, end reason)
        self.answers: queue.Queue = queue.Queue()   # subject -> harness: payload, or the error

    def reset(self):
        return self.rows.get()

    def step(self, outputs):
        self.answers.put(outputs[self.channel])
        return self.rows.get()


def multi_agent_rollout(subjects: Union[Sequence, Dict[str, Any]],
                        mediator: Mediator, n_turns: int,
                        order: Optional[List[str]] = None,
                        outputs: Optional[Sequence[str]] = None,
                        action_channel: str = 'behavior:text') -> List[Dict[str, Any]]:
    """Run a turn-taking conversation: every turn, each subject in order gets
    its row from the mediator and answers on ``action_channel``; the mediator
    routes the answer into the row of the next subject.

    ``outputs`` are the channels each session requests (default
    ``(action_channel,)``); every subject must declare them in ``out_channels``.
    Returns a transcript: one ``{'turn', 'subject', 'action'}`` per answer, in
    order, where ``action`` is the payload on ``action_channel``. An entry also
    holds each other requested output of that turn under its channel name. The
    shared state is on ``mediator`` (for example ``mediator.history``).
    """
    if isinstance(subjects, dict):
        subjects = [subjects[k] for k in (order or list(subjects))]
    subjects = list(subjects)
    if not subjects:
        raise ValueError('multi_agent_rollout needs at least one subject')
    if len({id(s) for s in subjects}) != len(subjects):
        raise ValueError('A subject takes part in one session at a time; give each seat its own subject.')
    outputs = (action_channel,) if outputs is None else tuple(outputs)
    if action_channel not in outputs:
        raise ValueError(f'outputs must include the action channel {action_channel!r}; got {list(outputs)}')
    for subject in subjects:
        missing = sorted(set(outputs) - set(subject.out_channels))
        if missing:
            raise ValueError(f'Subject {subject.identifier!r} does not declare {missing} in out_channels. '
                             f"A policy answers there with policy_out_channels=('{action_channel}',).")

    seats = [_Seat(action_channel) for _ in subjects]
    sessions = [environment_session(seat, outputs, action_channel=action_channel) for seat in seats]
    threads = [threading.Thread(target=contextvars.copy_context().run,
                                args=(_take_part, subject, session, seat),
                                name=f'participate-{subject.identifier}', daemon=True)
               for subject, session, seat in zip(subjects, sessions, seats)]
    for thread in threads:
        thread.start()

    transcript: List[Dict[str, Any]] = []
    turns: List[List[Dict[str, Any]]] = [[] for _ in subjects]   # the entries of each subject
    last: Optional[StreamEvent] = None
    reason = 'terminal'
    try:
        for turn in range(n_turns):
            for subject, seat, entries in zip(subjects, seats, turns):
                seat.rows.put((dict(mediator.observe(subject.identifier, last)), None))
                answer = seat.answers.get()
                if isinstance(answer, BaseException):
                    raise answer
                last = StreamEvent(action_channel, answer, None, {'sender': subject.identifier})
                mediator.apply(subject.identifier, last)
                entries.append({'turn': turn, 'subject': subject.identifier, 'action': answer})
                transcript.append(entries[-1])
    except BaseException:
        reason = 'failure'
        raise
    finally:
        for seat in seats:
            seat.rows.put(({}, reason))
        for thread in threads:
            thread.join()
    for entries, session in zip(turns, sessions):
        _add_recordings(entries, session, action_channel)
    return transcript


def _take_part(subject, session, seat: _Seat) -> None:
    """Run the subject's session; give the harness any error, or say the session ended."""
    try:
        with session:
            subject.participate(session)
        seat.answers.put(RuntimeError(
            f'The session of {subject.identifier!r} ended before its turn. '
            "Declare 'control' in in_channels and answer on the respond signal."))
    except BaseException as error:
        seat.answers.put(error)


def _add_recordings(entries, session: Session, action_channel: str) -> None:
    """Put a subject's other outputs in the entry of the turn they were sent in.

    A session keeps the subject's responses in order: the other outputs of a
    turn, then its answer on ``action_channel``.
    """
    turn, pending = 0, {}
    for event in session.emitted:
        if event.channel == action_channel:
            entries[turn].update(pending)
            turn, pending = turn + 1, {}
        else:
            pending[event.channel] = event.payload


class EchoAgent(Subject):
    """A subject that repeats what it heard. No network; for examples and tests.

    It shows the whole subject contract: it declares its channels and takes
    part through ``participate``. On each ``respond`` signal it answers on
    ``behavior:text`` with the ``text`` of that turn (``None`` when the turn
    had none).
    """
    in_channels = {'text', 'control'}
    out_channels = {'behavior:text'}

    def __init__(self, identifier: str):
        self._identifier = identifier

    @property
    def identifier(self) -> str:
        return self._identifier

    def participate(self, session: Session) -> None:
        session.declare_inputs(self.in_channels)
        heard = None
        while (event := session.next_input()) is not None:
            if event.channel == 'text':
                heard = event.payload
            elif event.channel == 'control' and event.payload == 'respond':
                session.respond(StreamEvent('behavior:text', f'{self.identifier}: heard<{heard}>',
                                            event.t_ms))
                heard = None
