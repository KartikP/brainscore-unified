"""Capability extension contract for BrainScoreModel."""


class Capability:
    """One ability of a BrainScoreModel: it runs the sessions it claims.

    ``supports_session`` claims the outputs a session requests, before any
    input is read; ``participate`` runs that session. A model gets an added
    capability from its constructor, ``BrainScoreModel(capabilities=[...])``.
    ``subject.participate(session)`` selects one capability for each session:
    an added capability that claims it runs it, before a built-in one; two
    claimants of the same kind (two added, or two built-in) are an error.

    Every method takes the subject (the ``BrainScoreModel``) first. Settings
    go to the capability's own constructor. A capability keeps its state on
    itself: give each model its own instance. ``reset`` runs at
    ``subject.reset()``.
    """

    identifier: str
    input_channels = frozenset()
    output_channels = frozenset()

    def declared_input_channels(self, subject):
        """Input channels this capability adds for ``subject``."""
        del subject
        return set(self.input_channels)

    def declared_output_channels(self, subject):
        """Output channels this capability adds for ``subject``."""
        del subject
        return set(self.output_channels)

    def reset(self, subject) -> None:
        """Release experiment state. Retain configuration needed by later calls."""
        del subject

    def supports_session(self, subject, channels) -> bool:
        """Claim a complete requested output combination before any input is read."""
        del subject, channels
        return False

    def participate(self, subject, session) -> None:
        """Run a session this capability claimed with ``supports_session``."""
        raise NotImplementedError(self.identifier)
