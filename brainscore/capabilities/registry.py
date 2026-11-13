"""The one capability registry of BrainScoreModel.

``BrainScoreModel.check_session_support`` reads it to find the capability that
runs a session.
"""

from typing import Dict

from .base import Capability


capability_registry: Dict[str, Capability] = {}


def register_capability(capability: Capability, *, replace: bool = False) -> Capability:
    """Register ``capability`` by identifier and return it."""
    if not getattr(capability, 'identifier', None):
        raise ValueError("Capability must define a non-empty identifier.")
    previous = capability_registry.get(capability.identifier)
    if previous is not None and previous is not capability and not replace:
        raise ValueError(f"Capability {capability.identifier!r} already registered; use replace=True explicitly.")
    capability_registry[capability.identifier] = capability
    return capability


def _setup_once(subject, capability: Capability) -> None:
    """Run ``capability.setup`` for ``subject`` once, before the capability first runs."""
    if capability.identifier in subject._capability_setup_done:
        return
    state = capability.setup(subject)
    subject._capability_state[capability.identifier] = (
        state if state is not None else {}
    )
    subject._capability_setup_done.add(capability.identifier)
