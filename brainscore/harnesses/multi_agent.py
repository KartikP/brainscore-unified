"""Multi-agent rollout harness: connect two (or more) subjects so each one's
output becomes the next agent's observation.

The unified interface gives the per-agent tick: ``process(row) -> outputs`` via
each agent's ``policy``, where a row is a mapping of channel -> payload. This
harness owns the *loop* and the *routing*: a :class:`Mediator` turns one agent's
output into the next agent's row and holds the shared state (a conversation
transcript, a gridworld, a market book, ...). That output-to-observation routing
is the whole substance of "two agents interacting"; everything else is per-agent
dispatch the interface already provides.

    mediator = DialogueMediator(task='agree on a meeting time')
    transcript = multi_agent_rollout([alice, bob], mediator, n_turns=8)

A message between agents is text: a ``StreamEvent`` on ``text`` with the sender
in ``meta['sender']``.

To study the interaction, hook each agent's network with
:class:`brainscore.activation_window.ActivationWindow` for the duration of the
rollout: it fires on every forward pass this loop drives, giving time-aligned
per-tick activations from both agents over a shared interaction. The loop
calls ``process``, not ``participate``, so a session tool (``observed``,
``Witness``) sees nothing here.

Any object with ``.identifier`` and ``.process(row) -> outputs`` is a valid agent
(a BrainScoreModel with a ``policy``, or any Subject). The harness is duck-typed
and device-agnostic; it stays out of ``core``, like the other harnesses.
"""
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from brainscore_core.streaming import StreamEvent


class Mediator:
    """Routes agent outputs into the next agent's row and holds shared state.
    Subclass for a specific world.

    ``observe(agent_id, incoming)`` builds the row the agent perceives this tick
    (``incoming`` is the previous agent's output event, or ``None`` on the first
    tick). ``apply(agent_id, event)`` folds the agent's output event into the
    shared state.
    """
    task: Optional[str] = None

    def observe(self, agent_id: str, incoming: Optional[StreamEvent]) -> Mapping[str, Any]:
        raise NotImplementedError

    def apply(self, agent_id: str, event: StreamEvent) -> None:
        raise NotImplementedError


class DialogueMediator(Mediator):
    """A turn-taking conversation. Each agent observes the running transcript and
    the peer's last utterance; its output is appended as the next utterance.

    The utterance rides inside ``observation`` as plain text in a dict. For a
    message on the ``text`` channel with the sender in its meta, use
    :class:`MessageMediator`.
    """

    def __init__(self, task: Optional[str] = None):
        self.task = task
        self.history: List[Dict[str, Any]] = []

    def observe(self, agent_id: str, incoming: Optional[StreamEvent]) -> Dict[str, Any]:
        return {'observation': {'instruction': self.task,
                                'incoming': None if incoming is None else incoming.payload,
                                'history': list(self.history)},
                'instruction': self.task}

    def apply(self, agent_id: str, event: StreamEvent) -> None:
        self.history.append({'agent': agent_id, 'utterance': event.payload})


class MessageMediator(Mediator):
    """Agent-to-agent messaging on the ``text`` channel.

    Each agent receives the peer's last message as ``row['text']`` (or the task as
    a seed message on the first turn); ``row['observation']`` names the sender,
    the recipient and the history. The shared transcript is a list of ``text``
    events on ``self.history``, each with ``meta['sender']``.
    """

    def __init__(self, task: Optional[str] = None):
        self.task = task
        self.history: List[StreamEvent] = []

    def observe(self, agent_id: str, incoming: Optional[StreamEvent]) -> Dict[str, Any]:
        content = self.task if incoming is None else incoming.payload
        sender = 'task' if incoming is None else incoming.meta.get('sender')
        return {'text': content,
                'observation': {'sender': sender, 'recipient': agent_id, 'task': self.task,
                                'history': [m.payload for m in self.history]}}

    def apply(self, agent_id: str, event: StreamEvent) -> None:
        self.history.append(event)


def _agent_id(agent) -> str:
    return getattr(agent, 'identifier', getattr(agent, '_identifier', repr(agent)))


def multi_agent_rollout(agents: Union[Sequence, Dict[str, Any]],
                        mediator: Mediator, n_turns: int,
                        order: Optional[List[str]] = None,
                        channel: str = 'text') -> List[Dict[str, Any]]:
    """Run a turn-taking rollout: every turn, each agent observes (via the
    mediator) -> ``process(row)`` -> its output on ``channel`` is routed back
    through the mediator into the next agent's row.

    Returns a transcript: a list of ``{turn, agent, action}`` in firing order,
    where ``action`` is the payload sent on ``channel``. The shared state lives on
    ``mediator`` (e.g. ``mediator.history``).
    """
    if isinstance(agents, dict):
        agent_list = [agents[k] for k in (order or list(agents.keys()))]
    else:
        agent_list = list(agents)
    if not agent_list:
        raise ValueError("multi_agent_rollout needs at least one agent")

    transcript: List[Dict[str, Any]] = []
    last: Optional[StreamEvent] = None
    for t in range(n_turns):
        for agent in agent_list:
            aid = _agent_id(agent)
            outputs = agent.process(mediator.observe(aid, last))
            value = outputs[channel]
            event = value if isinstance(value, StreamEvent) else StreamEvent(channel, value, None)
            event = replace(event, meta={'sender': aid, **event.meta})
            mediator.apply(aid, event)
            last = event
            transcript.append({'turn': t, 'agent': aid, 'action': event.payload})
    return transcript


class EchoAgent:
    """Trivial demo agent that echoes what it heard. No weights, for examples/tests.

    Shows the agent contract: ``.identifier`` + ``.process(row) -> outputs``. A real
    agent is a BrainScoreModel whose ``policy`` runs a forward pass (which an
    ActivationWindow then captures)."""

    def __init__(self, identifier: str):
        self.identifier = identifier

    def process(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        observation = row.get('observation')
        incoming = observation.get('incoming') if isinstance(observation, dict) else None
        return {'text': f'{self.identifier}: heard<{incoming}>'}


class MessageEchoAgent:
    """Trivial message demo agent: reads ``row['text']`` and answers on ``text``.
    Pairs with :class:`MessageMediator`."""

    def __init__(self, identifier: str):
        self.identifier = identifier

    def process(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        return {'text': StreamEvent('text', f"{self.identifier}: heard<{row['text']}>", None,
                                    {'sender': self.identifier})}
