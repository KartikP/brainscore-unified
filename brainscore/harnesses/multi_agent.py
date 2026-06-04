"""Multi-agent rollout harness — connect two (or more) subjects so each one's
output becomes the next agent's observation.

The unified interface gives the per-agent tick: ``process(EnvironmentStep) ->
EnvironmentResponse`` via each agent's ``action_fn``. This harness owns the
*loop* and the *routing*: a :class:`Mediator` turns one agent's action into the
next agent's observation and holds the shared state (a conversation transcript,
a gridworld, a market book, …). That output→observation routing is the whole
substance of "two agents interacting"; everything else is per-agent dispatch the
interface already provides.

    mediator = DialogueMediator(task='agree on a meeting time')
    transcript = multi_agent_rollout([alice, bob], mediator, n_turns=8)

To study the interaction, wrap each agent in :class:`brainscore.witness.Witness`
(what each saw/did per tick) and :class:`brainscore.activation_window.Activation
Window` (per-tick hidden activations) for the duration of the rollout — both
fire on the same ``process()`` calls this loop drives, giving time-aligned
traces from both agents over a shared interaction.

Any object with ``.identifier`` and ``.process(EnvironmentStep) ->
EnvironmentResponse`` is a valid agent (a BrainScoreModel with an ``action_fn``,
or any UnifiedModel). The harness is duck-typed and device-agnostic — it stays
out of ``core``, like the other harnesses.
"""
from typing import Any, Dict, List, Optional, Sequence, Union

from brainscore_core.model_interface import EnvironmentStep, EnvironmentResponse


class Mediator:
    """Routes agent outputs into the next agent's observation and holds shared
    state. Subclass for a specific world.

    ``observe(agent_id, incoming)`` builds the observation the agent perceives
    this tick (``incoming`` is the previous agent's action, or ``None`` on the
    first tick). ``apply(agent_id, action)`` folds the agent's action into the
    shared state.
    """
    task: Optional[str] = None

    def observe(self, agent_id: str, incoming: Any) -> Any:
        raise NotImplementedError

    def apply(self, agent_id: str, action: Any) -> None:
        raise NotImplementedError


class DialogueMediator(Mediator):
    """A turn-taking conversation. Each agent observes the running transcript and
    the peer's last utterance; its action is appended as the next utterance.

    This is the shared-world mediator in its simplest form — direct, typed
    agent-to-agent messaging (a first-class communicative ``Message`` that is
    valid as both output and input) is the OutputEvent-symmetry roadmap item;
    until then the utterance rides inside ``observation`` as plain text here.
    """

    def __init__(self, task: Optional[str] = None):
        self.task = task
        self.history: List[Dict[str, Any]] = []

    def observe(self, agent_id: str, incoming: Any) -> Dict[str, Any]:
        return {'instruction': self.task, 'incoming': incoming,
                'history': list(self.history)}

    def apply(self, agent_id: str, action: Any) -> None:
        self.history.append({'agent': agent_id, 'utterance': action})


def _agent_id(agent) -> str:
    return getattr(agent, 'identifier', getattr(agent, '_identifier', repr(agent)))


def multi_agent_rollout(agents: Union[Sequence, Dict[str, Any]],
                        mediator: Mediator, n_turns: int,
                        order: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Run a turn-taking rollout: every turn, each agent observes (via the
    mediator) → ``process(EnvironmentStep)`` → its action is routed back through
    the mediator into the next agent's observation.

    Returns a transcript: a list of ``{turn, agent, action}`` in firing order.
    The shared state lives on ``mediator`` (e.g. ``mediator.history``).
    """
    if isinstance(agents, dict):
        agent_list = [agents[k] for k in (order or list(agents.keys()))]
    else:
        agent_list = list(agents)
    if not agent_list:
        raise ValueError("multi_agent_rollout needs at least one agent")

    transcript: List[Dict[str, Any]] = []
    last_action: Any = None
    for t in range(n_turns):
        for agent in agent_list:
            aid = _agent_id(agent)
            obs = mediator.observe(aid, last_action)
            step = EnvironmentStep(observation=obs, step_num=t,
                                   instruction=getattr(mediator, 'task', None))
            resp = agent.process(step)
            action = resp.action if isinstance(resp, EnvironmentResponse) else resp
            mediator.apply(aid, action)
            last_action = action
            transcript.append({'turn': t, 'agent': aid, 'action': action})
    return transcript


class EchoAgent:
    """Trivial demo agent — echoes what it heard. No weights, for examples/tests.

    Shows the agent contract: ``.identifier`` + ``.process(EnvironmentStep) ->
    EnvironmentResponse``. A real agent is a BrainScoreModel whose ``action_fn``
    runs a forward pass (which an ActivationWindow then captures)."""

    def __init__(self, identifier: str):
        self.identifier = identifier

    def process(self, step: EnvironmentStep) -> EnvironmentResponse:
        incoming = step.observation.get('incoming') if isinstance(step.observation, dict) else None
        return EnvironmentResponse(action=f'{self.identifier}: heard<{incoming}>')
