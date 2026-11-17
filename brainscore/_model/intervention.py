"""The active interventions of a BrainScoreModel and their undo."""

from contextlib import contextmanager
from typing import Callable, Dict

from brainscore_core.events import PerturbationApplied
from ..selection import require_resolved


class Interventions:
    """Owns the active interventions of one BrainScoreModel instance.

    An intervention arrives as a ``StateChange`` on a declared channel
    (``lesion``, ``stimulation`` or ``pharmacological``), sent by a session;
    ``apply_state_change`` and the ``Intervene`` tool build that session.
    The intervention session part gives this object to the session driver,
    which calls ``process`` with each change.
    """

    def __init__(self, owner) -> None:
        self.owner = owner
        self.active_interventions: Dict[str, Callable[[], None]] = {}
        self._intact_depth = 0      # open ``intact()`` blocks

    def _refuse_while_intact(self, action: str) -> None:
        if self._intact_depth:
            raise RuntimeError(
                f"Subject '{self.owner.identifier}' cannot {action} while its interventions "
                f"are switched off (inside intact()). Change the interventions before or "
                f"after the block.")

    @contextmanager
    def intact(self):
        """Switch every active change off for the block, then on again.

        Each change stays in place: its hook keeps its place among the other
        hooks of the network, and its handle and cleanup do not change. Inside
        the block the network is intact, so an extraction may use the
        activation cache of the intact model. Only a change made by
        ``pytorch_intervention`` has a switch; with any other change
        active, the block refuses before it changes anything.

        The set of interventions is fixed while the block runs: applying,
        undoing or resetting an intervention raises ``RuntimeError``, so the
        exit switches on exactly the changes that were active at entry. Blocks
        nest; only the outermost one switches. The exit also runs when the
        block raises.
        """
        if self._intact_depth:
            self._intact_depth += 1
            try:
                yield
            finally:
                self._intact_depth -= 1
            return
        from brainscore.model_helpers.intervention import _SWITCHES
        switches = [(handle, _SWITCHES.get(cleanup))
                    for handle, cleanup in self.active_interventions.items()]
        fixed = [handle for handle, switch in switches if switch is None]
        if fixed:
            from brainscore_core.compatibility import ExecutionNotFeasibleError
            raise ExecutionNotFeasibleError(
                f"Subject '{self.owner.identifier}' cannot fit on the intact network: the "
                f"active intervention(s) {fixed} cannot be switched off in place. Only "
                f"pytorch_intervention makes changes that can. Use "
                f"readout_fit='current', or fit the readout before the change.")
        off = []
        self._intact_depth = 1
        try:
            for handle, switch in reversed(switches):
                switch(False)
                off.append((handle, switch))
            yield
        finally:
            self._intact_depth = 0
            failures = []
            for handle, switch in reversed(off):
                try:
                    switch(True)
                except Exception as error:                   # noqa: BLE001
                    failures.append(f'{handle!r}: {type(error).__name__}: {error}')
            if failures:
                raise RuntimeError(
                    f"{len(failures)} intervention(s) could not be switched on again, so "
                    f"the network is not as the active interventions say. The handles "
                    f"remain registered: {'; '.join(failures)}")

    def reset(self) -> None:
        """Undo every active intervention.

        One cleanup that raises must not prevent the others from running, so
        each is attempted independently. A failed cleanup is *not* forgotten,
        though: its handle stays registered, and once every cleanup was
        attempted the reset raises ``RuntimeError`` naming the retained
        handles. A cleanup that raised leaves the model still changed, so a
        normal return would let every later score be computed on a damaged
        model. Refused inside ``intact()``.
        """
        self._refuse_while_intact('reset its interventions')
        failures = []
        for handle_id, cleanup in list(self.active_interventions.items()):
            try:
                cleanup()
            except Exception as error:                       # noqa: BLE001
                failures.append((handle_id, error))
            else:
                self.active_interventions.pop(handle_id, None)
        if failures:
            detail = '; '.join(f'{handle!r}: {type(err).__name__}: {err}'
                               for handle, err in failures)
            raise RuntimeError(
                f"{len(failures)} intervention cleanup(s) failed, so this model "
                f"is still changed and any score taken from it now is invalid. "
                f"The handles remain registered and reset() can be retried: "
                f"{detail}") from failures[0][1]

    def process(self, state_change):
        """Apply a StateChange with the subject's ``intervention`` part, or undo one by handle.

        The subject changes under the readout, so the next ``start_task`` fits
        again, even for the task already configured. Refused inside ``intact()``.
        """
        owner = self.owner
        self._refuse_while_intact('apply or undo an intervention')
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
            # The model changes from here on: the readout fits again next time.
            # A refused event (above) leaves it as it was.
            owner._behavioral.invalidate()
            # Deregister only once the undo has actually succeeded; popping
            # first would lose the handle to a raising cleanup and leave the
            # model changed with no way to retry.
            cleanup()
            self.active_interventions.pop(handle_id, None)
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
        owner._behavioral.invalidate()      # the model changes from here on (a refused event did not)
        result = owner._intervention(state_change)
        # The part has already run, so a rejected return may leave the model
        # changed: undo it with the returned cleanup, if there is one.
        cleanup = (result[1] if isinstance(result, tuple) and len(result) == 2
                   and callable(result[1]) else None)
        if cleanup is None or not isinstance(result[0], PerturbationApplied):
            self._reject(cleanup, TypeError(
                f"Subject '{owner.identifier}' intervention must return a "
                f"(PerturbationApplied, cleanup_callable) tuple. Got "
                f"{self._returned(result)}."
            ))
        applied = result[0]
        if applied.handle_id in self.active_interventions:
            self._reject(cleanup, ValueError(
                f"Duplicate handle_id {applied.handle_id!r} from the intervention part. "
                f"An intervention must return a new handle_id for every call."
            ))
        self.active_interventions[applied.handle_id] = cleanup
        return applied

    @staticmethod
    def _returned(result) -> str:
        """Describe a rejected return: the tuple's wrong part, or the type of anything else."""
        if isinstance(result, tuple) and len(result) == 2:
            if not isinstance(result[0], PerturbationApplied):
                return f"a tuple whose first element is {type(result[0]).__name__}"
            return f"a tuple whose second element is {type(result[1]).__name__}"
        return type(result).__name__

    def _reject(self, cleanup, error: Exception) -> None:
        """Raise ``error`` for a rejected intervention return, after undoing it if possible.

        The cleanups of a part are assumed independent of each other, so undoing
        this one call leaves the earlier tracked changes as they are.
        """
        reload = (f" The intervention part of subject '{self.owner.identifier}' already ran, "
                  f"so the model may be changed; reload the subject before any further use.")
        if cleanup is None:
            raise type(error)(f"{error}{reload}")
        try:
            cleanup()
        except Exception as cleanup_error:                   # noqa: BLE001
            raise type(error)(
                f"{error} Its cleanup also failed ({type(cleanup_error).__name__}: "
                f"{cleanup_error}).{reload}") from cleanup_error
        raise error


