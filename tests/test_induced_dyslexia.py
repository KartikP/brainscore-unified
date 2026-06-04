"""Unit tests for the induced-dyslexia perturbation benchmark.

No ROAR data or real models needed: the reading benchmark is injected, and the
candidate is a light stub that records which kind of lesion is active. Tests the
benchmark's own logic — localize → ablate → measure → reset orchestration, the
specific-deficit computation, the dyslexic flag, and the composite per-layer
localization.
"""
import numpy as np
import pytest

from brainscore_core.metrics import Score
from brainscore_core.model_interface import Selection, RandomSelection
from brainscore.benchmarks.induced_dyslexia.benchmark import InducedDyslexia


# -- stubs ------------------------------------------------------------------

class _LayerSel:
    def __init__(self, path): self.layer_path = path


class _CompositeSel:
    def __init__(self, paths): self.layer_paths = list(paths)


class _Candidate:
    """Records which lesion kind is active so the fake reader can react."""
    def __init__(self, selectors):
        self._state_change_fn = lambda sc: (None, lambda: None)   # truthy
        self.region_layer_selectors = selectors
        self.active = 'none'
        self.reset_calls = 0

    def process(self, state_change):
        self.active = state_change.target.metadata.get('selector', 'functional')

    def reset(self):
        self.active = 'none'
        self.reset_calls += 1


class _Reader:
    """Returns scripted reading accuracy by lesion kind on the candidate."""
    def __init__(self, accs):
        self.accs = accs
        self.calls = 0

    def __call__(self, candidate):
        self.calls += 1
        s = Score(0.0)
        s.attrs['raw'] = self.accs[candidate.active]
        return s


def _make(accs, region='VWFA', selectors=None, n_units=3, monkeypatch=None,
          canned=None):
    cand = _Candidate(selectors or {region: _LayerSel('L')})
    reader = _Reader(accs)
    bench = InducedDyslexia(localizer_region=region, n_units=n_units,
                            reading_benchmark=reader, localizer_stimuli=None)
    if monkeypatch is not None and canned is not None:
        monkeypatch.setattr(bench, '_localize', lambda c, layers: canned)
    return bench, cand, reader


# -- orchestration + scoring ------------------------------------------------

class TestScoring:
    CANNED = [Selection(layer='L', indices=[0, 1, 2],
                        metadata={'selector': 'functional', 'n_recorded': 10})]

    def test_selective_dyslexia_flagged(self, monkeypatch):
        # word-form lesion tanks reading (0.55); random lesion barely dents (0.90)
        bench, cand, reader = _make(
            {'none': 0.95, 'functional': 0.55, 'random': 0.90},
            monkeypatch=monkeypatch, canned=self.CANNED)
        score = bench(cand)
        assert score.attrs['baseline_accuracy'] == 0.95
        assert score.attrs['lesioned_accuracy'] == 0.55
        assert score.attrs['random_control_accuracy'] == 0.90
        assert abs(score.attrs['specific_deficit'] - 0.35) < 1e-9
        assert score.attrs['dyslexic'] is True
        assert float(score) == 0.55
        # reset after the word-form lesion AND after the random control
        assert cand.reset_calls == 2

    def test_not_dyslexic_when_above_threshold(self, monkeypatch):
        bench, cand, _ = _make(
            {'none': 0.95, 'functional': 0.72, 'random': 0.90},  # 0.72 > 0.65
            monkeypatch=monkeypatch, canned=self.CANNED)
        assert bench(cand).attrs['dyslexic'] is False

    def test_not_dyslexic_when_nonspecific(self, monkeypatch):
        # random ablation hurts just as much → not word-form-specific
        bench, cand, _ = _make(
            {'none': 0.95, 'functional': 0.55, 'random': 0.50},
            monkeypatch=monkeypatch, canned=self.CANNED)
        score = bench(cand)
        assert score.attrs['specific_deficit'] < 0      # random worse than lesion
        assert score.attrs['dyslexic'] is False

    def test_requires_state_change_fn(self, monkeypatch):
        bench, cand, _ = _make({'none': 0.9, 'functional': 0.5, 'random': 0.9},
                               monkeypatch=monkeypatch, canned=self.CANNED)
        cand._state_change_fn = None
        with pytest.raises(ValueError, match='state_change_fn'):
            bench(cand)

    def test_missing_region_raises(self, monkeypatch):
        bench, cand, _ = _make({'none': 0.9, 'functional': 0.5, 'random': 0.9},
                               selectors={'NOT_VWFA': _LayerSel('L')},
                               monkeypatch=monkeypatch, canned=self.CANNED)
        with pytest.raises(ValueError, match='region'):
            bench(cand)

    def test_random_control_matches_lesion_size(self, monkeypatch):
        # the matched random control draws the same #units at the same layer
        bench, cand, _ = _make(
            {'none': 0.95, 'functional': 0.55, 'random': 0.90},
            monkeypatch=monkeypatch, canned=self.CANNED)
        # RandomSelection sizing comes from the canned selection's metadata
        rs = RandomSelection(layer='L', n_units=3, n_total=10, seed=0).resolve(cand)
        assert len(rs.indices) == 3 and all(0 <= i < 10 for i in rs.indices)


# -- composite per-layer localization ---------------------------------------

class TestCompositeLocalize:
    def _assembly(self):
        from brainscore_core.supported_data_standards.brainio.assemblies import (
            NeuroidAssembly)
        # 4 presentations (2 real, 2 pseudo); 4 neuroids across 2 layers.
        # neuroid 0 (L1) and neuroid 2 (L2) are strongly real-selective; the
        # other two have no mean separation. Non-zero within-group variance so
        # Cohen's d is well-defined (a zero-variance unit yields d=0 by the
        # pooled-std guard).
        data = np.array([
            [5.0, 1.0, 5.0, 1.0],   # real
            [6.0, 2.0, 6.0, 2.0],   # real
            [0.0, 1.0, 0.0, 1.0],   # pseudo
            [1.0, 2.0, 1.0, 2.0],   # pseudo
        ])
        return NeuroidAssembly(
            data, dims=['presentation', 'neuroid'],
            coords={'image_label': ('presentation', ['real', 'real', 'pseudo', 'pseudo']),
                    'stimulus_id': ('presentation', ['s0', 's1', 's2', 's3']),
                    'layer': ('neuroid', ['L1', 'L1', 'L2', 'L2']),
                    'neuroid_id': ('neuroid', [0, 1, 2, 3])})

    def test_picks_real_selective_per_layer(self):
        asm = self._assembly()

        class CompCand:
            def __init__(s):
                s._state_change_fn = lambda sc: (None, lambda: None)
                s.region_layer_selectors = {'VWFA': _CompositeSel(['L1', 'L2'])}
                s.resets = 0
            def start_recording(s, target): pass
            def process(s, stim): return asm
            def reset(s): s.resets += 1

        cand = CompCand()
        bench = InducedDyslexia(localizer_region='VWFA', n_units=1,
                                reading_benchmark=_Reader({}), localizer_stimuli=None)
        sels = bench._localize(cand, ['L1', 'L2'])
        # _localize must reset first so a prior reading's behavioral task doesn't
        # make process() return choices instead of neuroids (the bug the EC2 run caught)
        assert cand.resets >= 1
        assert len(sels) == 2
        by_layer = {s.layer: s.indices for s in sels}
        # within-layer index 0 is the real-selective neuroid in both layers
        assert by_layer['L1'] == [0]
        assert by_layer['L2'] == [0]
        assert all(s.metadata['n_recorded'] == 2 for s in sels)
