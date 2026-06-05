"""End-to-end scoring-orchestration test (CI-able, no real weights/data).

Exercises the full pipeline glue with a SYNTHETIC model + benchmark:
    score() -> load_model -> load_benchmark -> benchmark(model)
              -> model.start_recording -> model.process(StimulusSet)
              -> NeuroidAssembly -> metric -> Score (+ runtime_sec/identifiers).

This is the production-blocking gap the coverage audit flagged: components were
tested in isolation but the score() loop never was. Uses a tiny seeded
feature extractor (duck-typed like TextWrapper: has `.identifier`, callable as
``(stimuli, layers=[...]) -> NeuroidAssembly``) so it runs offline in CI.
"""
import numpy as np
import pandas as pd
import pytest

import brainscore
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

N_FEATURES = 8


class _SyntheticExtractor:
    """Duck-typed extractor: deterministic random features per stimulus.
    Has `.identifier` so BrainScoreModel.process() calls it as a full
    extractor — (stimuli, layers=[...]) -> NeuroidAssembly."""
    def __init__(self, identifier='synthetic-extractor'):
        self.identifier = identifier

    def __call__(self, stimuli, layers):
        sentences = list(stimuli['sentence'].values)
        ids = [str(s) for s in stimuli['stimulus_id'].values]
        n = len(sentences)
        layer = layers[0]
        # deterministic per-sentence features (hash-seeded) so the test is stable
        feats = np.stack([
            np.random.RandomState(abs(hash(s)) % (2**32)).randn(N_FEATURES)
            for s in sentences]).astype(np.float32)
        return NeuroidAssembly(
            feats,
            coords={
                # ≥2 presentation coords so brainio's gather_indexes builds a
                # proper MultiIndex and exposes 'stimulus_id' as a level (xarray
                # 2022.3 quirk — a lone coord stays buried in .indexes).
                'stimulus_id': ('presentation', ids),
                'stimulus_num': ('presentation', list(range(n))),
                'neuroid_id': ('neuroid', [f'{layer}.{i}' for i in range(N_FEATURES)]),
                'neuroid_num': ('neuroid', list(range(N_FEATURES))),
                'layer': ('neuroid', [layer] * N_FEATURES),
            },
            dims=['presentation', 'neuroid'],
        )


class _SyntheticBenchmark:
    """Minimal benchmark: records a region, processes the stimuli through the
    model, and scores assembly-variance as a trivial deterministic metric."""
    identifier = 'synthetic-benchmark'
    required_modalities = {'text'}

    def __init__(self, stimuli):
        self._stimuli = stimuli

    def __call__(self, model):
        model.start_recording('IT')
        assembly = model.process(self._stimuli)
        assert assembly.dims == ('presentation', 'neuroid')
        # trivial, deterministic "alignment" metric in [0,1]
        value = float(np.tanh(np.abs(assembly.values).mean()))
        return Score(value)


@pytest.fixture
def stimuli():
    ss = StimulusSet(pd.DataFrame({
        'sentence': ['the cat sat', 'a dog ran', 'birds fly high', 'fish swim deep'],
        'stimulus_id': ['s0', 's1', 's2', 's3'],
    }))
    ss.identifier = 'synthetic-stim'
    return ss


@pytest.fixture
def model():
    return BrainScoreModel(
        identifier='synthetic-model', model=None,
        region_layer_map={'IT': 'layer.0'},
        preprocessors={'text': _SyntheticExtractor()},
    )


@pytest.mark.integration
class TestScoringOrchestration:
    def test_process_dispatches_to_preprocessor(self, model, stimuli):
        model.start_recording('IT')
        asm = model.process(stimuli)
        assert asm.dims == ('presentation', 'neuroid')
        assert asm.shape == (4, N_FEATURES)
        assert set(str(s) for s in asm['stimulus_id'].values) == {'s0', 's1', 's2', 's3'}

    def test_benchmark_end_to_end(self, model, stimuli):
        score = _SyntheticBenchmark(stimuli)(model)
        assert isinstance(score, Score)
        assert 0.0 <= float(score) <= 1.0

    def test_score_loop_attaches_runtime_and_identifiers(self, model, stimuli, monkeypatch):
        """The full score() orchestration: registry lookup -> benchmark(model)
        -> Score with runtime_sec + identifiers attached."""
        bench = _SyntheticBenchmark(stimuli)
        monkeypatch.setattr(brainscore, 'load_model', lambda i: model)
        monkeypatch.setattr(brainscore, 'load_benchmark', lambda i: bench)
        s = brainscore.score('synthetic-model', 'synthetic-benchmark')
        assert 'runtime_sec' in s.attrs and s.attrs['runtime_sec'] >= 0.0
        assert s.attrs['model_identifier'] == 'synthetic-model'
        assert s.attrs['benchmark_identifier'] == 'synthetic-benchmark'
        assert 0.0 <= float(s) <= 1.0

    def test_reproducible_across_runs(self, model, stimuli):
        """Deterministic extractor -> identical score on re-run (no flakiness)."""
        model.start_recording('IT')
        a = model.process(stimuli).values
        b = model.process(stimuli).values
        assert np.allclose(a, b)
