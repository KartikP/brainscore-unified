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

Threads: ``participate`` returns only when its session ends, and every session
stays open for the whole conversation, so each one runs in its own thread.
Only one subject acts at a time, so a run is as deterministic as its subjects.

The outputs of a turn are what the subject sends between the ``respond`` signal
of that turn and the next one, in any order. A turn that misses a requested
output, or an error of any subject, stops the run with an error.
"""
import contextvars
import queue
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from brainscore_core.contract import Subject
from brainscore_core.streaming import CONTROL_CHANNEL, RESPOND, Session, StreamEvent
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
    ``turns`` holds each turn's outputs (channel to payload), from the session
    events; ``error`` is what the subject's thread raised, if anything.
    """

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.rows: queue.Queue = queue.Queue()      # harness -> subject: (row, end reason)
        self.answers: queue.Queue = queue.Queue()   # subject -> harness: payload, or _ENDED
        self.turns: List[Dict[str, Any]] = []
        self.duplicate: Optional[tuple] = None      # (turn, channel) answered twice
        self.error: Optional[BaseException] = None

    def reset(self):
        return self.rows.get()

    def step(self, outputs):
        self.answers.put(outputs[self.channel])
        return self.rows.get()

    def on_input(self, event) -> None:
        if isinstance(event, StreamEvent) and event.channel == CONTROL_CHANNEL and event.payload == RESPOND:
            self.turns.append({})

    def on_output(self, event: StreamEvent) -> None:
        turn = len(self.turns) - 1
        if turn < 0 or event.channel in self.turns[turn]:
            self.duplicate = self.duplicate or (turn, event.channel)
            return
        self.turns[turn][event.channel] = event.payload


_ENDED = object()   # put on ``answers`` when a subject's thread ends


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

    order_of_answers: List[tuple] = []     # (turn, seat index) in the order the answers came
    last: Optional[StreamEvent] = None
    reason = 'terminal'
    try:
        for turn in range(n_turns):
            for index, (subject, seat) in enumerate(zip(subjects, seats)):
                seat.rows.put((dict(mediator.observe(subject.identifier, last)), None))
                answer = seat.answers.get()
                if answer is _ENDED:
                    raise seat.error or RuntimeError(
                        f'The session of {subject.identifier!r} ended before its turn {turn}. '
                        "Declare 'control' in in_channels and answer on the respond signal.")
                last = StreamEvent(action_channel, answer, None, {'sender': subject.identifier})
                mediator.apply(subject.identifier, last)
                order_of_answers.append((turn, index))
    except BaseException:
        reason = 'failure'
        raise
    finally:
        for seat in seats:
            seat.rows.put(({}, reason))
        for thread in threads:
            thread.join()
    for subject, seat in zip(subjects, seats):
        if seat.error is not None:
            raise seat.error
        _check_turns(subject.identifier, seat, n_turns, outputs)
    return [{'turn': turn, 'subject': subjects[index].identifier,
             'action': seats[index].turns[turn][action_channel],
             **{channel: payload for channel, payload in seats[index].turns[turn].items()
                if channel != action_channel}}
            for turn, index in order_of_answers]


def _take_part(subject, session, seat: _Seat) -> None:
    """Run the subject's session; keep its error on the seat, then wake a waiting harness."""
    try:
        with session, session.observe(on_input=seat.on_input, on_output=seat.on_output):
            subject.participate(session)
    except BaseException as error:
        seat.error = error
    finally:
        seat.answers.put(_ENDED)


def _check_turns(identifier: str, seat: _Seat, n_turns: int, outputs: Sequence[str]) -> None:
    """Every turn holds each requested output once."""
    if seat.duplicate is not None:
        turn, channel = seat.duplicate
        where = 'before its first turn' if turn < 0 else f'twice in turn {turn}'
        raise ValueError(f'Subject {identifier!r} answered on {channel!r} {where}.')
    for turn in range(n_turns):
        given = seat.turns[turn] if turn < len(seat.turns) else {}
        for channel in outputs:
            if channel not in given:
                raise ValueError(f'Subject {identifier!r} sent no {channel!r} in turn {turn}.')


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
