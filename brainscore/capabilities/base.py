"""Capability extension contract for BrainScoreModel."""

from typing import Any, Dict


class Capability:
    """One ability of a BrainScoreModel, with one role.

    A capability runs sessions, or it answers events; it does not do both.

    - A session capability claims the outputs a session requests
      (``supports_session``) and runs that session (``participate``);
      ``subject.participate(session)`` reaches it. A capability added from
      another package normally takes this role.
    - An event capability claims one event (``handles``) and answers it
      (``process``). The built-in session capabilities reach these through the
      session drivers, which call the subject's ``process``.

    Implement the pair for the role; the other pair declines by default. One
    registry holds both roles and tries them in ``order``, lower first, ties by
    ``identifier``. The built-in event capabilities have orders 50 to 1000 and
    the built-in session capabilities 10000, so an added capability (default
    1000) is tried before a built-in one that claims the same outputs.

    Every method takes the subject (the ``BrainScoreModel``) first. Per-subject
    state comes from ``setup`` and is read back with
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

        The registry stores it before the capability first runs; read it back
        with ``subject.capability_state(self.identifier)``.
        """
        del subject
        return {}

    def reset(self, subject, state) -> None:
        """Release experiment state. Retain configuration needed by later calls."""
        del subject
        state.clear()

    # --- session role ---------------------------------------------------

    def supports_session(self, subject, channels) -> bool:
        """Claim a complete requested output combination before any input is read."""
        del subject, channels
        return False

    def participate(self, subject, session) -> None:
        """Run a session this capability claimed with ``supports_session``."""
        raise NotImplementedError(self.identifier)

    # --- event role -----------------------------------------------------

    def handles(self, subject, event, **kwargs) -> bool:
        """Claim ``event``, which the subject's ``process`` received."""
        del subject, event, kwargs
        return False

    def process(self, subject, event, **kwargs):
        """Answer an event this capability claimed with ``handles``."""
        raise NotImplementedError(self.identifier)
