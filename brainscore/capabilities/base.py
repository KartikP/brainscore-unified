"""Capability extension contract for BrainScoreModel."""

from typing import Any, Dict


class Capability:
    """One ability of a BrainScoreModel: it runs the sessions it claims.

    ``supports_session`` claims the outputs a session requests, before any
    input is read; ``participate`` runs that session.
    ``subject.participate(session)`` selects one capability for each session:
    the claimant with the lowest ``order``; two claimants at that order are an
    error. The built-in session parts have order 10000, so an added capability
    (default 1000) runs a session that a built-in one also claims.

    Every method takes the subject (the ``BrainScoreModel``) first. Per-subject
    state comes from ``setup``, which runs once before the capability first
    runs a session (again after ``subject.reset()``), and is read back with
    ``subject.capability_state(identifier)``; settings come from
    ``subject.capability_config``.
    """

    identifier: str
    order: int = 1000
    input_channels = frozenset()
    output_channels = frozenset()

    def enabled_for(self, subject) -> bool:
        """Whether this capability is considered for ``subject``."""
        del subject
        return True

    def declared_input_channels(self, subject):
        """Input channels this capability adds for ``subject``."""
        del subject
        return set(self.input_channels)

    def declared_output_channels(self, subject):
        """Output channels this capability adds for ``subject``."""
        del subject
        return set(self.output_channels)

    def setup(self, subject) -> Dict[str, Any]:
        """Return this capability's initial per-subject state.

        It is stored before the capability first runs a session; read it back
        with ``subject.capability_state(self.identifier)``.
        """
        del subject
        return {}

    def reset(self, subject, state) -> None:
        """Release experiment state. Retain configuration needed by later calls."""
        del subject
        state.clear()

    def supports_session(self, subject, channels) -> bool:
        """Claim a complete requested output combination before any input is read."""
        del subject, channels
        return False

    def participate(self, subject, session) -> None:
        """Run a session this capability claimed with ``supports_session``."""
        raise NotImplementedError(self.identifier)
