"""Tests for interventions on BrainScoreModel, sent with ``apply_state_change``.

Covers:
- the intervention part receives the change through the intervention session
- A subject without an intervention part is refused
- Bad return type raises TypeError
- handle_id collision raises ValueError
- model.reset() runs all cleanups and clears the registry
- StateChange(kind='reset', handle_id=...) selectively undoes
- A simple synthetic ablation flow: baseline → apply → confirm side-effect →
  reset → side-effect gone
"""

import pytest

from brainscore import BrainScoreModel
from brainscore_core.compatibility import CompatibilityError
from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
from brainscore_core.streaming_helpers import apply_state_change


# ── Helpers for dispatch tests ───────────────────────────────────────

def _make_model(intervention=None):
    """Minimal BrainScoreModel for intervention testing."""
    return BrainScoreModel(
        identifier='test-state-change',
        model=None,
        region_layer_map={},
        in_channels={'vision:image'},
        intervention=intervention,
    )


def _counting_state_change_fn():
    """Returns a state_change_fn that records every apply/cleanup call.

    The closure exposes `state['installed']` (set of handle_ids currently
    installed) so tests can verify cleanup runs.
    """
    state = {'next_id': 0, 'installed': set()}

    def fn(state_change: StateChange):
        handle_id = f"handle-{state['next_id']}"
        state['next_id'] += 1
        state['installed'].add(handle_id)

        applied = PerturbationApplied(
            handle_id=handle_id,
            target=state_change.target,
            perturbation=state_change.perturbation,
            applied_at=float(state['next_id']),
        )

        def cleanup():
            state['installed'].discard(handle_id)

        return applied, cleanup

    return fn, state


# ── Dispatch ─────────────────────────────────────────────────────────

class TestInterventionDispatch:

    def test_subject_without_intervention_part_is_refused(self):
        model = _make_model(intervention=None)
        sc = StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        )
        with pytest.raises(CompatibilityError, match='no intervention part'):
            apply_state_change(model, sc)

    def test_apply_returns_perturbation_applied(self):
        fn, state = _counting_state_change_fn()
        model = _make_model(intervention=fn)
        sc = StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10', indices=[1, 2, 3]),
            perturbation=Perturbation(kind='zero'),
        )
        applied = apply_state_change(model, sc)
        assert isinstance(applied, PerturbationApplied)
        assert applied.target.indices == [1, 2, 3]
        assert applied.handle_id in state['installed']

    def test_input_site_and_vector_amount_reach_state_change_fn_unchanged(self):
        """Core does not interpret ':input' or the amount's type; the
        registered state_change_fn receives exactly what was declared."""
        import numpy as np
        fn, _ = _counting_state_change_fn()
        model = _make_model(intervention=fn)
        direction = np.arange(4.0)
        applied = apply_state_change(model, StateChange(
            kind='steer',
            target=Selection(layer='model.layers.14.mlp.down_proj:input'),
            perturbation=Perturbation(kind='drive', amount=direction),
        ))
        assert applied.target.layer == 'model.layers.14.mlp.down_proj:input'
        assert applied.perturbation.amount is direction

    def test_state_change_fn_must_return_tuple(self):
        def bad_fn(state_change):
            return PerturbationApplied(
                handle_id='x',
                target=state_change.target,
                perturbation=state_change.perturbation,
            )  # forgot the cleanup callable

        model = _make_model(intervention=bad_fn)
        sc = StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        )
        with pytest.raises(TypeError, match='must return a'):
            apply_state_change(model, sc)

    def test_state_change_fn_must_return_callable_cleanup(self):
        def bad_fn(state_change):
            applied = PerturbationApplied(
                handle_id='x',
                target=state_change.target,
                perturbation=state_change.perturbation,
            )
            return applied, 'not callable'  # second arg not callable

        model = _make_model(intervention=bad_fn)
        sc = StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        )
        with pytest.raises(TypeError, match='must return a'):
            apply_state_change(model, sc)

    def test_duplicate_handle_id_raises(self):
        # state_change_fn that always returns the same handle_id
        def colliding_fn(state_change):
            applied = PerturbationApplied(
                handle_id='same-id',
                target=state_change.target,
                perturbation=state_change.perturbation,
            )
            return applied, lambda: None

        model = _make_model(intervention=colliding_fn)
        sc = StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        )
        apply_state_change(model, sc)  # first one installs fine
        with pytest.raises(ValueError, match='Duplicate handle_id'):
            apply_state_change(model, sc)


# ── Reset semantics ──────────────────────────────────────────────────

class TestResetClearsPerturbations:

    def test_reset_runs_all_cleanups(self):
        fn, state = _counting_state_change_fn()
        model = _make_model(intervention=fn)
        for _ in range(3):
            apply_state_change(model, StateChange(
                kind='ablation',
                target=Selection(layer='blocks.10'),
                perturbation=Perturbation(kind='zero'),
            ))
        assert len(state['installed']) == 3
        model.reset()
        assert state['installed'] == set()
        with pytest.raises(KeyError, match='No active intervention'):   # nothing left to undo
            apply_state_change(model, StateChange(kind='reset', handle_id='handle-0'))

    def test_reset_tolerates_failing_cleanup(self):
        # If one cleanup raises, the others should still run.
        cleanup_log = []
        counter = [0]

        def fn_raising(state_change):
            handle_id = f"h-{counter[0]}"
            counter[0] += 1

            def cleanup():
                if handle_id == 'h-1':
                    raise RuntimeError("cleanup failed")
                cleanup_log.append(handle_id)

            applied = PerturbationApplied(
                handle_id=handle_id,
                target=state_change.target,
                perturbation=state_change.perturbation,
            )
            return applied, cleanup

        model = _make_model(intervention=fn_raising)
        for _ in range(3):
            apply_state_change(model, StateChange(
                kind='ablation',
                target=Selection(layer='blocks.10'),
                perturbation=Perturbation(kind='zero'),
            ))
        with pytest.warns(RuntimeWarning, match='still changed'):
            model.reset()  # should NOT raise
        # h-0 and h-2 ran cleanly; h-1 raised but was still tolerated
        assert 'h-0' in cleanup_log and 'h-2' in cleanup_log
        # h-1's cleanup failed, so that perturbation is still in effect. Its
        # handle is deliberately retained: forgetting it would leave the model
        # damaged while reporting nothing active. Only the succeeding two are
        # deregistered.
        with pytest.raises(KeyError, match='No active intervention'):
            apply_state_change(model, StateChange(kind='reset', handle_id='h-0'))
        with pytest.raises(RuntimeError, match='cleanup failed'):    # h-1 is still known
            apply_state_change(model, StateChange(kind='reset', handle_id='h-1'))


class TestSelectiveReset:
    """StateChange(kind='reset', handle_id=X) undoes just one perturbation."""

    def test_kind_reset_removes_one(self):
        fn, state = _counting_state_change_fn()
        model = _make_model(intervention=fn)
        a = apply_state_change(model, StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        ))
        b = apply_state_change(model, StateChange(
            kind='ablation',
            target=Selection(layer='blocks.20'),
            perturbation=Perturbation(kind='zero'),
        ))
        assert state['installed'] == {a.handle_id, b.handle_id}

        apply_state_change(model, StateChange(kind='reset', handle_id=a.handle_id))
        assert state['installed'] == {b.handle_id}
        with pytest.raises(KeyError, match='No active intervention'):   # a is gone
            apply_state_change(model, StateChange(kind='reset', handle_id=a.handle_id))
        apply_state_change(model, StateChange(kind='reset', handle_id=b.handle_id))  # b stays
        assert state['installed'] == set()

    def test_kind_reset_unknown_handle_raises(self):
        fn, _ = _counting_state_change_fn()
        model = _make_model(intervention=fn)
        with pytest.raises(KeyError, match='No active intervention'):
            apply_state_change(model, StateChange(kind='reset', handle_id='nonexistent'))

    def test_kind_reset_missing_handle_id_raises(self):
        fn, _ = _counting_state_change_fn()
        model = _make_model(intervention=fn)
        with pytest.raises(ValueError, match='requires handle_id'):
            apply_state_change(model, StateChange(kind='reset', handle_id=None))


# ── Synthetic ablation lifecycle ─────────────────────────────────────

class TestSyntheticAblationLifecycle:
    """End-to-end: install an ablation, observe a side effect on a fake
    'forward pass', reset, observe the side effect is gone. No torch — uses
    a plain dict to stand in for a model's hidden state."""

    def test_baseline_apply_observe_reset_observe(self):
        # Fake "model state" that the perturbation will mutate
        model_state = {'output': 1.0}

        def state_change_fn(state_change):
            # Save current value so cleanup can restore
            saved = model_state['output']
            # Apply the ablation: zero out
            if state_change.perturbation.kind == 'zero':
                model_state['output'] = 0.0
            elif state_change.perturbation.kind == 'scale':
                model_state['output'] *= state_change.perturbation.scale

            applied = PerturbationApplied(
                handle_id=f"ablation-{id(state_change)}",
                target=state_change.target,
                perturbation=state_change.perturbation,
            )

            def cleanup():
                model_state['output'] = saved

            return applied, cleanup

        model = _make_model(intervention=state_change_fn)
        # Baseline
        assert model_state['output'] == 1.0
        # Apply ablation
        applied = apply_state_change(model, StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        ))
        # Observe perturbed state
        assert model_state['output'] == 0.0
        # Reset
        model.reset()
        # Side effect undone
        assert model_state['output'] == 1.0
        with pytest.raises(KeyError, match='No active intervention'):
            apply_state_change(model, StateChange(kind='reset', handle_id=applied.handle_id))


class TestCleanupFailuresAreNotLost:
    """A cleanup that raises leaves the model perturbed.

    Forgetting the handle in that case is worse than the failure itself: the
    model reports nothing active while still being damaged, so every later
    score is silently taken from a lesioned model.
    """

    @staticmethod
    def _model_with(cleanups):
        """A model whose intervention part installs the given cleanups, one per change."""
        pending = list(cleanups.items())

        def intervention(state_change):
            handle_id, cleanup = pending.pop(0)
            return PerturbationApplied(handle_id=handle_id, target=state_change.target,
                                       perturbation=state_change.perturbation), cleanup

        model = _make_model(intervention=intervention)
        for _ in cleanups:
            apply_state_change(model, StateChange(kind='ablation', target=Selection(layer='blocks.10'),
                                                  perturbation=Perturbation(kind='zero')))
        return model

    @staticmethod
    def _undo(model, handle_id):
        apply_state_change(model, StateChange(kind='reset', handle_id=handle_id))

    def test_failed_cleanup_keeps_its_handle_and_others_still_run(self):
        ran = []

        def boom():
            raise RuntimeError('hook already detached')

        model = self._model_with({'bad': boom, 'good': lambda: ran.append('good')})
        with pytest.warns(RuntimeWarning, match='still changed'):
            model.reset()
        assert ran == ['good']
        with pytest.raises(KeyError):                     # good is gone
            self._undo(model, 'good')
        with pytest.raises(RuntimeError, match='detached'):   # bad stays registered
            self._undo(model, 'bad')

    def test_clean_reset_is_silent_and_empties_the_registry(self):
        import warnings
        model = self._model_with({'a': lambda: None, 'b': lambda: None})
        with warnings.catch_warnings():
            warnings.simplefilter('error')       # any warning fails the test
            model.reset()
        for handle_id in ('a', 'b'):
            with pytest.raises(KeyError):
                self._undo(model, handle_id)

    def test_retrying_reset_clears_a_handle_that_later_succeeds(self):
        import warnings
        state = {'fail': True}

        def flaky():
            if state['fail']:
                state['fail'] = False
                raise RuntimeError('transient')

        model = self._model_with({'flaky': flaky})
        with pytest.warns(RuntimeWarning):
            model.reset()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            model.reset()                        # second attempt succeeds, silently
        with pytest.raises(KeyError):
            self._undo(model, 'flaky')

    def test_selective_reset_keeps_the_handle_when_undo_raises(self):
        def boom():
            raise RuntimeError('nope')

        model = self._model_with({'h1': boom})
        with pytest.raises(RuntimeError):
            self._undo(model, 'h1')
        with pytest.raises(RuntimeError, match='nope'):   # still registered, so tried again
            self._undo(model, 'h1')
