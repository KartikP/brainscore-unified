"""score() hands legacy domain benchmarks the permanent facade."""
import numpy as np
import pandas as pd

import brainscore
from brainscore_core.metrics import Score
from brainscore import BrainScoreModel
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


class _Extractor:
    def __call__(self, stimuli, layers=None, **kwargs):
        n = len(stimuli)
        return NeuroidAssembly(
            np.arange(n * 2, dtype=float).reshape(n, 2),
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'neuroid_id': ('neuroid', [0, 1]), 'layer': ('neuroid', [layers[0]] * 2)},
            dims=['presentation', 'neuroid'])


class _LegacyBenchmark:
    identifier = 'legacy-vision'
    region = 'IT'
    required_modalities = {'vision'}
    subject_api = 'legacy'  # what a domain loader records for a legacy benchmark

    def __init__(self):
        self.stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b'],
                                                 'image_path': ['/tmp/a.png', '/tmp/b.png']}))
        self.stimuli.identifier = 'two-images'

    def __call__(self, candidate):
        candidate.start_recording('IT', time_bins=[(70, 170)])
        output = candidate.look_at(self.stimuli, number_of_trials=1)
        score = Score([1.0], coords={'aggregation': ['center']}, dims=['aggregation'])
        score.attrs.update(candidate=type(candidate).__name__, degrees=candidate.visual_degrees(),
                           output=output)
        return score


class _TypedBenchmark(_LegacyBenchmark):
    identifier = 'typed-vision'
    subject_api = 'typed'

    def __call__(self, candidate):
        candidate.start_recording('IT', time_bins=[(70, 170)])
        output = candidate.process(self.stimuli)
        score = Score([1.0], coords={'aggregation': ['center']}, dims=['aggregation'])
        score.attrs.update(candidate=type(candidate).__name__, output=output)
        return score


def test_score_wraps_a_legacy_benchmark_and_not_a_typed_one():
    model = BrainScoreModel('typed', region_layer_map={'IT': 'layer'},
                            extractor={'vision': _Extractor()}, visual_degrees=8)
    legacy = brainscore.score(model, _LegacyBenchmark(), check_mem=False)
    typed = brainscore.score(model, _TypedBenchmark(), check_mem=False)
    assert legacy.attrs['candidate'] == 'LegacyBenchmarkFacade'
    assert legacy.attrs['degrees'] == 8
    assert typed.attrs['candidate'] == 'BrainScoreModel'
    np.testing.assert_array_equal(legacy.attrs['output'].values, typed.attrs['output'].values)
    assert legacy.attrs['compatibility']['match'] == 'full'


class _VisionNeuralBenchmark(_LegacyBenchmark):
    """What a legacy vision NeuralBenchmark does to its candidate."""

    def preallocate_memory(self, *args, **kwargs):
        pass


def test_typed_model_scores_identically_to_process(monkeypatch):
    import brainscore_vision

    model = BrainScoreModel('typed', region_layer_map={'IT': 'layer'},
                            extractor={'vision': _Extractor()}, visual_degrees=8)
    benchmark = _VisionNeuralBenchmark()
    monkeypatch.setattr(brainscore_vision, 'import_plugin', lambda *a, **k: None)
    monkeypatch.setitem(brainscore_vision.benchmark_registry, 'legacy-test', lambda: benchmark)
    monkeypatch.setattr(brainscore_vision, 'load_model', lambda _: model)
    score = brainscore_vision._run_score('m', 'legacy-test', check_mem=False)
    assert score.attrs['candidate'] == 'LegacyBenchmarkFacade'
    assert score.attrs['degrees'] == 8
    model.start_recording('IT', time_bins=[(70, 170)])
    np.testing.assert_array_equal(score.attrs['output'].values, model.process(benchmark.stimuli).values)
