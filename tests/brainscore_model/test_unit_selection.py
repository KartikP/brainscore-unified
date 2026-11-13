"""UnitSelection family: portable, resolve(subject)-based perturbation selection.

The functional localizer (word-vs-non-word, Cohen's d ranking) and its
random-unit control. Unit tests run against a participate-only fake subject (a
synthetic assembly, no model weights); integration tests send changes with
``apply_state_change`` to prove that an unresolved UnitSelection target is
refused with the fix in the message, and that a resolved FunctionalSelection
reaches the intervention part as the exact units.
"""
import numpy as np
import pytest

from brainscore import BrainScoreModel, UnitSelection, FunctionalSelection, RandomSelection, IndexSelection
from brainscore_core import Subject
from brainscore_core.model_interface import Selection, Perturbation, PerturbationApplied, StateChange
from brainscore_core.streaming import StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


# ── Fake Subject that returns a synthetic localizer assembly ─────────────

LABELS = ['word'] * 4 + ['nonword'] * 4


def _localizer_assembly(layer='enc.5', n_neuro=10, discriminating=5, layers=None):
    """8 presentations (4 'word', 4 'nonword'); the first `discriminating`
    units respond strongly to 'word' (large Cohen's d), the rest have no group
    difference (d == 0, deterministically — not noise that might spike)."""
    word = np.array([l == 'word' for l in LABELS])
    rng = np.random.RandomState(0)
    data = np.zeros((8, n_neuro))
    # discriminating units: small within-group noise (so pooled std > 0) + a
    # strong +3 word bump -> d is large and finite
    data[:, :discriminating] = rng.normal(0, 0.1, (8, discriminating))
    data[np.ix_(word, np.arange(discriminating))] += 3.0
    # non-discriminating units: constant per unit -> identical across groups -> d == 0
    for j in range(discriminating, n_neuro):
        data[:, j] = float(j)
    return NeuroidAssembly(
        data, dims=['presentation', 'neuroid'],
        coords={'stimulus_id': ('presentation', [f's{i}' for i in range(8)]),
                'stimulus_index': ('presentation', list(range(8))),
                'neuroid_id': ('neuroid', list(range(n_neuro))),
                'layer': ('neuroid', layers or [layer] * n_neuro)})


def _localizer_stimuli():
    stimuli = StimulusSet({'stimulus_id': [f's{i}' for i in range(8)],
                           'sentence': [f'w{i}' for i in range(8)], 'label': LABELS})
    stimuli.identifier = 'unit-selection-localizer'
    return stimuli


class _FakeSubject(Subject):
    """A participate-only subject: it answers a neural session with a canned assembly."""
    identifier = 'fake-localizer-subject'
    in_channels = {'text'}
    out_channels = {'neural:VWFA'}

    def __init__(self, assembly):
        self._asm = assembly
        self.recorded = None

    def participate(self, session):
        (channel,) = session.outputs
        self.recorded = channel
        while session.next_input() is not None:
            pass
        session.respond(StreamEvent(channel, self._asm, 0.0, {}))

    def reset(self):
        pass


def _functional(**kwargs):
    settings = dict(region='VWFA', localizer_stimuli=_localizer_stimuli(),
                    contrast=(['word'], ['nonword']))
    settings.update(kwargs)
    return FunctionalSelection(**settings)


# ── FunctionalSelection ──────────────────────────────────────────────

class TestFunctionalSelection:

    def test_positive_picks_discriminating_units(self):
        model = _FakeSubject(_localizer_assembly())
        sel = _functional(n_units=5, sign='positive').resolve(model)
        assert isinstance(sel, Selection)
        assert sel.indices == [0, 1, 2, 3, 4]
        assert sel.layer == 'enc.5'                  # read from the assembly's layer coord
        assert sel.metadata['selector'] == 'functional'
        assert sel.metadata['n_recorded'] == 10
        assert model.recorded == 'neural:VWFA'       # recorded the requested region

    def test_negative_picks_the_complement(self):
        model = _FakeSubject(_localizer_assembly())
        sel = _functional(n_units=5, sign='negative').resolve(model)
        assert sel.indices == [5, 6, 7, 8, 9]

    def test_abs_matches_positive_here(self):
        model = _FakeSubject(_localizer_assembly())
        sel = _functional(n_units=5, sign='abs').resolve(model)
        assert sel.indices == [0, 1, 2, 3, 4]

    def test_threshold_mode_selects_above_cutoff(self):
        model = _FakeSubject(_localizer_assembly())
        sel = _functional(n_units=None, threshold=2.0).resolve(model)
        assert sel.indices == [0, 1, 2, 3, 4]         # only the strong units clear d>=2

    def test_empty_contrast_group_raises(self):
        model = _FakeSubject(_localizer_assembly())
        with pytest.raises(ValueError, match='non-empty'):
            _functional(contrast=(['absent_label'], ['nonword'])).resolve(model)

    def test_bad_sign_raises(self):
        model = _FakeSubject(_localizer_assembly())
        with pytest.raises(ValueError, match='sign must be'):
            _functional(sign='sideways').resolve(model)
        assert model.recorded is None                # refused before any session

    def test_multi_layer_recording_raises(self):
        asm = _localizer_assembly(layers=['a'] * 5 + ['b'] * 5)
        with pytest.raises(ValueError, match='spans 2 layers'):
            _functional(n_units=3).resolve(_FakeSubject(asm))

    def test_resolve_layers_gives_one_selection_per_layer(self):
        asm = _localizer_assembly(layers=['b'] * 3 + ['a'] * 7)
        selections = _functional(n_units=None, threshold=2.0).resolve_layers(_FakeSubject(asm))
        assert [s.layer for s in selections] == ['b', 'a']   # recording order
        # units 0-4 discriminate: all three of layer b, the first two of layer a
        assert [s.indices for s in selections] == [[0, 1, 2], [0, 1]]
        assert [s.metadata['n_recorded'] for s in selections] == [3, 7]

    def test_recording_without_layer_is_refused(self):
        asm = _localizer_assembly().reset_index('neuroid').drop_vars('layer')
        with pytest.raises(ValueError, match="no 'layer' coord"):
            _functional(n_units=1).resolve(_FakeSubject(asm))

    def test_labels_come_from_the_stimuli(self):
        """A subject need not send stimulus columns back; presentations may come in any order."""
        asm = _localizer_assembly().isel(presentation=[7, 0, 6, 1, 5, 2, 4, 3])
        sel = _functional(n_units=5).resolve(_FakeSubject(asm))
        assert sel.indices == [0, 1, 2, 3, 4]

    def test_collapses_extra_time_bin_dim(self):
        asm = _localizer_assembly().expand_dims({'time_bin': 3}).copy()
        sel = _functional(n_units=5).resolve(_FakeSubject(asm))
        assert sel.indices == [0, 1, 2, 3, 4]


# ── RandomSelection (the causal control) ──────────────────────────────

class TestRandomSelection:

    def test_deterministic_and_in_range(self):
        sel = RandomSelection(layer='enc.5', n_units=3, n_total=10, seed=0).resolve(None)
        assert isinstance(sel, Selection)
        assert len(sel.indices) == 3
        assert all(0 <= i < 10 for i in sel.indices)
        again = RandomSelection(layer='enc.5', n_units=3, n_total=10, seed=0).resolve(None)
        assert sel.indices == again.indices            # deterministic by seed

    def test_different_seed_differs(self):
        a = RandomSelection(layer='enc.5', n_units=5, n_total=20, seed=0).resolve(None).indices
        b = RandomSelection(layer='enc.5', n_units=5, n_total=20, seed=1).resolve(None).indices
        assert a != b

    def test_clamps_to_n_total(self):
        sel = RandomSelection(layer='enc.5', n_units=99, n_total=10, seed=0).resolve(None)
        assert len(sel.indices) == 10


# ── IndexSelection ────────────────────────────────────────────────────

def test_index_selection_passthrough():
    sel = IndexSelection(layer='enc.5', indices=[2, 7]).resolve(None)
    assert sel.layer == 'enc.5'
    assert sel.indices == [2, 7]
    assert sel.metadata['selector'] == 'index'


# ── Integration through apply_state_change ───────────────────────────

def _make_model(intervention=None, region_layer_map=None):
    return BrainScoreModel(
        identifier='test-unit-selection', model=None,
        region_layer_map=region_layer_map or {},
        supported_modalities={'vision'},
        intervention=intervention,
    )


def _capturing_intervention(captured):
    def fn(sc):
        captured['target'] = sc.target
        applied = PerturbationApplied(handle_id='h0', target=sc.target,
                                      perturbation=sc.perturbation)
        return applied, lambda: None
    return fn


def test_plain_selection_reaches_the_intervention_unchanged():
    from brainscore_core.streaming_helpers import apply_state_change
    captured = {}
    model = _make_model(intervention=_capturing_intervention(captured))
    sc = StateChange(kind='ablation',
                     target=Selection(layer='b', indices=[0]),
                     perturbation=Perturbation(kind='zero'))
    apply_state_change(model, sc)
    assert captured['target'].indices == [0]


@pytest.mark.parametrize('target', [
    IndexSelection(layer='blocks.10', indices=[1, 2, 3]),
    RandomSelection('blocks.10', n_units=2, n_total=8),
    FunctionalSelection('IT', None, (['word'], ['nonword'])),
])
def test_an_unresolved_selection_is_refused_with_the_fix(target):
    """The localizer is its own session, so the target is resolved first."""
    from brainscore_core.streaming_helpers import apply_state_change
    captured = {}
    model = _make_model(intervention=_capturing_intervention(captured))
    sc = StateChange(kind='ablation', target=target, perturbation=Perturbation(kind='zero'))
    with pytest.raises(TypeError, match=r'selection\.resolve\(subject\)'):
        apply_state_change(model, sc)
    assert captured == {}                      # the intervention part never ran


def test_intervene_refuses_an_unresolved_selection_before_any_run():
    from brainscore.experiments import Intervene
    from types import SimpleNamespace
    model = _make_model(intervention=_capturing_intervention({}))
    tool = Intervene(StateChange('ablation', target=IndexSelection('b', [0]),
                                 perturbation=Perturbation('zero')))
    with pytest.raises(TypeError, match=r'selection\.resolve\(subject\)'):
        tool.validate(SimpleNamespace(subject=model))


def test_a_resolved_functional_selection_is_applied_through_a_session():
    """Resolve (the localizer session), then apply (the intervention session)."""
    from brainscore_core.streaming_helpers import apply_state_change

    class _Extractor:
        identifier = 'localizer-extractor'

        def __call__(self, stimuli, layers):
            return _localizer_assembly(layer=layers[0])

    captured = {}
    model = BrainScoreModel('functional-then-intervene', region_layer_map={'IT': 'enc.5'},
                            extractor={'text': _Extractor()},
                            intervention=_capturing_intervention(captured))
    units = FunctionalSelection('IT', _localizer_stimuli(), (['word'], ['nonword']),
                                n_units=3).resolve(model)
    applied = apply_state_change(model, StateChange('ablation', target=units,
                                                    perturbation=Perturbation('zero')))
    assert captured['target'] is units
    apply_state_change(model, StateChange(kind='reset', handle_id=applied.handle_id))  # it is active
    assert units.layer == 'enc.5' and len(units.indices) == 3


@pytest.mark.parametrize('population', [[4, 4], [-1, 4], [1.5, 4]])
def test_random_selection_rejects_ambiguous_population(population):
    with pytest.raises(ValueError, match='indices'):
        RandomSelection('L', n_units=1, n_total=2, population=population).resolve(None)


def test_subset_without_original_addresses_is_rejected():
    from brainscore.selection import CompositeSelector
    model = _FakeSubject(_localizer_assembly(n_neuro=2, discriminating=1))
    model.region_layer_selectors = {'VWFA': CompositeSelector((('enc.5', (4, 1)),))}
    with pytest.raises(ValueError, match='original unit_index'):
        FunctionalSelection('VWFA', _localizer_stimuli(), (['word'], ['nonword']), n_units=1).resolve(model)


def test_matched_to_copies_the_functional_selection_settings():
    from brainscore_core.model_interface import Selection
    from brainscore.selection import RandomSelection
    population = list(range(10, 60))
    units = Selection(layer='features.12', indices=[11, 12, 13], axis=1, expected_shape=(512, 13, 13),
                      metadata={'selector': 'functional', 'unit_population': population})
    by_hand = RandomSelection('features.12', n_units=3, n_total=50, population=population,
                              seed=4, axis=1, expected_shape=(512, 13, 13)).resolve(None)
    matched = RandomSelection.matched_to(units, seed=4).resolve(None)
    assert matched.indices == by_hand.indices
    assert (matched.layer, matched.axis, matched.expected_shape) == ('features.12', 1, (512, 13, 13))


def test_matched_to_needs_a_population():
    import pytest
    from brainscore_core.model_interface import Selection
    from brainscore.selection import RandomSelection
    with pytest.raises(ValueError, match='unit population'):
        RandomSelection.matched_to(Selection(layer='x', indices=[0], metadata={}))
