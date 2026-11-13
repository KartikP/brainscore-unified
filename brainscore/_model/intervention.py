"""The active interventions of a BrainScoreModel and their undo."""

from contextlib import contextmanager
from typing import Callable, Dict

from brainscore_core.events import PerturbationApplied
from ..selection import require_resolved

# The catalog's intervention input channels; the target is in the payload (rule 3.1).
INTERVENTION_CHANNELS = ('lesion', 'stimulation', 'pharmacological')


class Interventions:
    """Owns the active interventions of one BrainScoreModel instance.

    An intervention arrives as a ``StateChange`` on a declared channel
    (``lesion``, ``stimulation`` or ``pharmacological``), sent by a session;
    ``apply_state_change`` and the ``Intervene`` tool build that session.
    """

    def __init__(self, owner) -> None:
        self.owner = owner
        self.active_interventions: Dict[str, Callable[[], None]] = {}
        self._changes: Dict[str, object] = {}   # handle -> resolved StateChange

    @contextmanager
    def intact(self):
        """Undo every active change for the block, then apply each again.

        Inside the block no change is active, so an extraction may use the
        activation cache of the intact model. Handles keep their identity, so a
        later reset by handle still works.
        """
        undone = dict(self.active_interventions)
        for handle in reversed(list(undone)):
            undone[handle]()
        self.active_interventions = {}
        try:
            yield
        finally:
            self.active_interventions = undone
            for handle in undone:
                _, cleanup = self.owner._intervention(self._changes[handle])
                self.active_interventions[handle] = cleanup

    def reset(self) -> None:
        """Undo every active intervention.

        One cleanup that raises must not prevent the others from running, so
        each is attempted independently. A failed cleanup is *not* forgotten,
        though: its handle stays registered and the failure is reported. A
        cleanup that raised leaves the model still changed, so dropping the
        handle silently would leave every later score computed on a damaged
        model with nothing left to show it.
        """
        failures = []
        for handle_id, cleanup in list(self.active_interventions.items()):
            try:
                cleanup()
            except Exception as error:                       # noqa: BLE001
                failures.append((handle_id, error))
            else:
                self.active_interventions.pop(handle_id, None)
                self._changes.pop(handle_id, None)
        if failures:
            import warnings
            detail = '; '.join(f'{handle!r}: {type(err).__name__}: {err}'
                               for handle, err in failures)
            warnings.warn(
                f"{len(failures)} intervention cleanup(s) failed, so this model "
                f"is still changed and any score taken from it now is invalid. "
                f"The handles remain registered and reset() can be retried: "
                f"{detail}",
                RuntimeWarning,
                stacklevel=2,
            )

    def dispatch_state_change(self, state_change):
        """Apply a StateChange with the subject's ``intervention`` part, or undo one by handle."""
        owner = self.owner
        if state_change.kind == 'reset':
            handle_id = state_change.handle_id
            if handle_id is None:
                raise ValueError(
                    f"StateChange(kind='reset') requires handle_id to identify "
                    f"which intervention to undo. Use model.reset() to undo all."
                )
            cleanup = self.active_interventions.get(handle_id)
            if cleanup is None:
                raise KeyError(
                    f"No active intervention with handle_id={handle_id!r}. "
                    f"Active: {list(self.active_interventions.keys())}."
                )
            # Deregister only once the undo has actually succeeded; popping
            # first would lose the handle to a raising cleanup and leave the
            # model changed with no way to retry.
            cleanup()
            self.active_interventions.pop(handle_id, None)
            self._changes.pop(handle_id, None)
            return None

        if owner._intervention is None:
            raise NotImplementedError(
                f"Subject '{owner.identifier}' has no intervention part. "
                f"An intervention needs BrainScoreModel(intervention=...), a "
                f"callable intervention(state_change) -> (PerturbationApplied, "
                f"undo_function). "
                f"Received: StateChange(kind={state_change.kind!r})."
            )

        require_resolved(state_change)
        result = owner._intervention(state_change)
        if (not isinstance(result, tuple) or len(result) != 2
                or not isinstance(result[0], PerturbationApplied)
                or not callable(result[1])):
            raise TypeError(
                f"Subject '{owner.identifier}' intervention must return a "
                f"(PerturbationApplied, cleanup_callable) tuple. Got "
                f"{type(result).__name__}."
            )
        applied, cleanup = result
        if applied.handle_id in self.active_interventions:
            raise ValueError(
                f"Duplicate handle_id {applied.handle_id!r} from the intervention part. "
                f"An intervention must return a new handle_id for every call."
            )
        self.active_interventions[applied.handle_id] = cleanup
        self._changes[applied.handle_id] = state_change
        return applied


