"""Unit tests for the induced-dyslexia perturbation benchmark.

No ROAR data or real models needed. Every candidate here is a contract subject:
it implements only ``participate(session)``, ``reset()`` and its declarations,
with no ``process`` and no private attributes. Tests the benchmark's own logic
(localize, ablate, measure, reset), the specific-deficit computation, the
dyslexic flag, the composite per-layer localization, and pre-flight N/A.
"""
import numpy as np
import pytest

from brainscore_core import Subject
from brainscore_core.compatibility import CompatibilityError, check_compatibility
from brainscore_core.metrics import Score
from brainscore_core.model_interface import PerturbationApplied, Selection, RandomSelection
from brainscore_core.streaming import StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.benchmarks.induced_dyslexia.benchmark import InducedDyslexia


# -- stubs ------------------------------------------------------------------

class _Subject(Subject):
    """A participate-only subject. It answers lesion sessions with a record and
    neural sessions with a canned assembly, and remembers which lesion is active
    so the fake reader can react."""
    identifier = 'participate-only-stub'
    in_channels = {'text', 'lesion'}
    out_channels = {'behavior', 'neural:VWFA', 'perturbation'}

    def __init__(self, recording=None):
        self.recording = recording
        self.active = 'none'
        self.sessions = 0

    def participate(self, session):
        self.sessions += 1
        requested = set(session.requested_output_channels)
        if requested == {'perturbation'}:
            change = session.next_input().payload
            self.active = change.target.metadata.get('selector', 'functional')
            session.respond(StreamEvent('perturbation', PerturbationApplied(
                handle_id=f'h{self.sessions}', target=change.target,
                perturbation=change.perturbation), 0.0, {}))
            return
        while session.next_input() is not None:
            pass
        session.respond(StreamEvent('neural:VWFA', self.recording, 0.0, {}))

    def reset(self):
        self.active = 'none'


class _Reader:
    """Returns scripted reading accuracy by lesion kind on the candidate."""
    def __init__(self, accs):
        self.accs = accs
        self.seen = []

    def __call__(self, candidate):
        self.seen.append(candidate.active)
        s = Score(0.0)
        s.attrs['raw'] = self.accs[candidate.active]
        return s


def _make(accs, monkeypatch, canned):
    cand = _Subject()
    reader = _Reader(accs)
    bench = InducedDyslexia(n_units=3, modality='text',
                            reading_benchmark=reader, localizer_stimuli=None)
    monkeypatch.setattr(bench, '_localize', lambda c: canned)
    return bench, cand, reader


# -- orchestration + scoring ------------------------------------------------

class TestScoring:
    CANNED = [Selection(layer='L', indices=[0, 1, 2],
                        metadata={'selector': 'functional', 'n_recorded': 10})]

    @pytest.mark.parametrize('failed_call', [1, 2, 3])
    def test_reading_failure_removes_lesions(self, monkeypatch, failed_call):
        bench, cand, reader = _make(
            {'none': 0.95, 'functional': 0.55, 'random': 0.90},
            monkeypatch, self.CANNED)
        read = bench._reading_accuracy
        def failing_read(candidate):
            if len(reader.seen) + 1 == failed_call:
                raise RuntimeError('synthetic reading failure')
            return read(candidate)
        monkeypatch.setattr(bench, '_reading_accuracy', failing_read)
        with pytest.raises(RuntimeError, match='synthetic reading failure'):
            bench(cand)
        assert cand.active == 'none'

    def test_partial_installation_failure_removes_prior_lesion(self, monkeypatch):
        bench, cand, _ = _make(
            {'none': 0.95, 'functional': 0.55, 'random': 0.90},
            monkeypatch, self.CANNED)
        def partially_install(candidate, selections):
            candidate.active = 'functional'
            raise RuntimeError('second hook failed')
        monkeypatch.setattr(bench, '_ablate', partially_install)
        with pytest.raises(RuntimeError, match='second hook failed'):
            bench(cand)
        assert cand.active == 'none'

    def test_localization_failure_leaves_subject_reset(self, monkeypatch):
        bench, cand, _ = _make({'none': 0.95}, monkeypatch, self.CANNED)
        def failing_localize(candidate):
            candidate.active = 'functional'
            raise RuntimeError('localizer failed')
        monkeypatch.setattr(bench, '_localize', failing_localize)
        with pytest.raises(RuntimeError, match='localizer failed'):
            bench(cand)
        assert cand.active == 'none'

    def test_selective_dyslexia_flagged(self, monkeypatch):
        # word-form lesion tanks reading (0.55); random lesion barely dents (0.90)
        bench, cand, reader = _make(
            {'none': 0.95, 'functional': 0.55, 'random': 0.90},
            monkeypatch, self.CANNED)
        score = bench(cand)
        assert score.attrs['baseline_accuracy'] == 0.95
        assert score.attrs['lesioned_accuracy'] == 0.55
        assert score.attrs['random_control_accuracy'] == 0.90
        assert abs(score.attrs['specific_deficit'] - 0.35) < 1e-9
        assert score.attrs['dyslexic'] is True
        assert float(score) == 0.55
        # intact, then lesioned, then random control; no lesion left afterwards
        assert reader.seen == ['none', 'functional', 'random']
        assert cand.active == 'none'

    def test_not_dyslexic_when_above_threshold(self, monkeypatch):
        bench, cand, _ = _make(
            {'none': 0.95, 'functional': 0.72, 'random': 0.90},  # 0.72 > 0.65
            monkeypatch, self.CANNED)
        assert bench(cand).attrs['dyslexic'] is False

    def test_not_dyslexic_when_nonspecific(self, monkeypatch):
        # random ablation hurts just as much → not word-form-specific
        bench, cand, _ = _make(
            {'none': 0.95, 'functional': 0.55, 'random': 0.50},
            monkeypatch, self.CANNED)
        score = bench(cand)
        assert score.attrs['specific_deficit'] < 0      # random worse than lesion
        assert score.attrs['dyslexic'] is False

    def test_random_control_matches_lesion_size(self):
        # the matched random control draws the same #units at the same layer
        rs = RandomSelection(layer='L', n_units=3, n_total=10, seed=0).resolve(None)
        assert len(rs.indices) == 3 and all(0 <= i < 10 for i in rs.indices)


# -- pre-flight -------------------------------------------------------------

class TestPreflight:
    def _bench(self):
        return InducedDyslexia(modality='text', reading_benchmark=_Reader({}),
                               localizer_stimuli=None)

    def test_declares_what_it_sends_and_needs(self):
        bench = self._bench()
        assert bench.required_input_channels == {'text', 'lesion'}
        assert bench.requested_output_channels == {'behavior', 'neural:VWFA', 'perturbation'}

    def test_compatible_subject_matches(self):
        match = check_compatibility(_Subject(), self._bench())
        assert match.consumed_inputs == {'text', 'lesion'}

    @pytest.mark.parametrize('missing', ['perturbation', 'neural:VWFA'])
    def test_subject_without_output_is_na_before_compute(self, missing):
        subject = _Subject()
        subject.out_channels = _Subject.out_channels - {missing}
        with pytest.raises(CompatibilityError, match=missing):
            check_compatibility(subject, self._bench())
        assert subject.sessions == 0


# -- composite per-layer localization ---------------------------------------

class TestCompositeLocalize:
    def _assembly(self, with_layer=True):
        # 4 presentations (2 real, 2 pseudo); 4 neuroids across 2 layers.
        # neuroid 0 (L1) and neuroid 2 (L2) are strongly real-selective; the
        # other two have no mean separation. Non-zero within-group variance so
        # Cohen's d is well-defined (a zero-variance unit yields d=0 by the
        # pooled-std guard). Presentations arrive in a different order from the
        # stimulus set, so labels must be joined by stimulus_id.
        data = np.array([
            [0.0, 1.0, 0.0, 1.0],   # s2 pseudo
            [5.0, 1.0, 5.0, 1.0],   # s0 real
            [1.0, 2.0, 1.0, 2.0],   # s3 pseudo
            [6.0, 2.0, 6.0, 2.0],   # s1 real
        ])
        return NeuroidAssembly(
            data, dims=['presentation', 'neuroid'],
            coords={'stimulus_id': ('presentation', ['s2', 's0', 's3', 's1']),
                    'stimulus_index': ('presentation', [2, 0, 3, 1]),
                    'neuroid_id': ('neuroid', [0, 1, 2, 3]),
                    **({'layer': ('neuroid', ['L1', 'L1', 'L2', 'L2'])} if with_layer else
                       {'neuroid_num': ('neuroid', [0, 1, 2, 3])})})

    def _stimuli(self):
        stimuli = StimulusSet({'stimulus_id': ['s0', 's1', 's2', 's3'],
                               'sentence': ['cat', 'dog', 'gat', 'pob'],
                               'image_label': ['real', 'real', 'pseudo', 'pseudo']})
        stimuli.identifier = 'dyslexia-localizer-synthetic'
        return stimuli

    def test_picks_real_selective_per_layer(self):
        cand = _Subject(recording=self._assembly())
        bench = InducedDyslexia(localizer_region='VWFA', n_units=1, modality='text',
                                reading_benchmark=_Reader({}),
                                localizer_stimuli=self._stimuli())
        sels = bench._localize(cand)
        assert len(sels) == 2
        by_layer = {s.layer: s.indices for s in sels}
        # within-layer index 0 is the real-selective neuroid in both layers
        assert by_layer['L1'] == [0]
        assert by_layer['L2'] == [0]
        assert all(s.metadata['n_recorded'] == 2 for s in sels)

    def test_recording_without_layer_is_refused_before_any_lesion(self):
        cand = _Subject(recording=self._assembly(with_layer=False))
        bench = InducedDyslexia(n_units=1, modality='text', reading_benchmark=_Reader({}),
                                localizer_stimuli=self._stimuli())
        with pytest.raises(ValueError, match="no 'layer' coord"):
            bench._localize(cand)
        assert cand.sessions == 1      # the recording only; no lesion was sent


# -- end to end: real ROAR reading test, participate-only subject -------------

class _ReaderSubject(Subject):
    """Reads ``real-N`` / ``pseudo-N`` strings through ten units in one layer.

    Units 0 and 1 carry word form; with both silenced the subject calls every
    string real. It speaks only through ``participate``: neural activity as one
    assembly with a ``layer`` coord, answers as one probability row per trial,
    lesions as records.
    """
    identifier = 'participate-only-reader'
    in_channels = {'text', 'lesion'}
    out_channels = {'behavior', 'neural:VWFA', 'perturbation'}

    def __init__(self):
        self.silenced = {}
        self.applied = []

    @staticmethod
    def _units(word):
        kind, number = word.split('-')
        units = np.sin((np.arange(10) + 1) * (int(number) + 1))
        units[:2] += 5.0 * (kind == 'real')
        return units

    def participate(self, session):
        requested = set(session.requested_output_channels)
        if requested == {'perturbation'}:
            change = session.next_input().payload
            handle = f'h{len(self.applied)}'
            self.applied.append(change.target)
            self.silenced[handle] = set(change.target.indices)
            session.respond(StreamEvent('perturbation', PerturbationApplied(
                handle, change.target, change.perturbation), 0.0, {}))
            return
        events = []
        while (event := session.next_input()) is not None:
            if event.channel == 'text' and event.meta.get('role') != 'fitting':
                events.append(event)
        if requested == {'neural:VWFA'}:
            rows = np.stack([self._units(event.payload) for event in events])
            session.respond(StreamEvent('neural:VWFA', NeuroidAssembly(
                rows, dims=['presentation', 'neuroid'],
                coords={'stimulus_id': ('presentation', [e.meta['stimulus_id'] for e in events]),
                        'stimulus_index': ('presentation', list(range(len(events)))),
                        'layer': ('neuroid', ['L'] * 10),
                        'neuroid_id': ('neuroid', list(range(10)))}), 0.0, {}))
            return
        silenced = set().union(*self.silenced.values())
        for event in events:
            real = event.payload.startswith('real') or {0, 1} <= silenced
            session.respond(StreamEvent('behavior', np.array([1.0, 0.0] if real else [0.0, 1.0]),
                                        event.t_ms, {'stimulus_id': event.meta['stimulus_id'],
                                                     'label_set': ['real', 'pseudo']}))

    def reset(self):
        self.silenced.clear()


@pytest.fixture
def synthetic_roar(monkeypatch):
    from brainscore.benchmarks.roar_yeatman2021 import benchmark as roar
    ids = [f's{i}' for i in range(500)]
    labels = ['real'] * 250 + ['pseudo'] * 250
    words = [f'{label}-{i}' for i, label in enumerate(labels)]
    stimuli = StimulusSet({'stimulus_id': ids, 'sentence': words, 'image_label': labels})
    stimuli.identifier = 'synthetic-roar-dyslexia'
    stimuli.stimulus_paths = {sid: sid for sid in ids}
    monkeypatch.setattr(roar, 'load_stimulus_set', lambda identifier: stimuli)
    monkeypatch.setattr(roar, 'load_dataset', lambda identifier: None)
    monkeypatch.setattr(roar, '_human_accuracy_on_stimuli', lambda *a, **k: 1.0)


def test_participate_only_subject_runs_the_benchmark(synthetic_roar):
    subject = _ReaderSubject()
    bench = InducedDyslexia(n_units=2, modality='text')
    check_compatibility(subject, bench)
    score = bench(subject)
    assert score.attrs['baseline_accuracy'] == 1.0
    assert score.attrs['lesioned_accuracy'] == 0.5        # calls every string real
    assert score.attrs['random_control_accuracy'] == 1.0  # seed 0 misses units 0 and 1
    assert score.attrs['dyslexic'] is True
    assert score.attrs['n_layers_ablated'] == 1
    assert not subject.silenced                           # left as found
    lesion, control = subject.applied
    assert (lesion.layer, lesion.indices) == ('L', [0, 1])
    assert lesion.metadata['unit_population'] == list(range(10))
    expected = RandomSelection(layer='L', n_units=2, n_total=10, seed=0,
                               population=list(range(10))).resolve(None)
    assert (control.layer, control.indices) == ('L', expected.indices)
    assert control.metadata['unit_population'] == list(range(10))
