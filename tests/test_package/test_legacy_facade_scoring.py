"""score() hands legacy domain benchmarks the permanent facade."""
import numpy as np
import pandas as pd

import brainscore
from brainscore_core.metrics import Score
from brainscore_core.streaming_helpers import neural_response
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


class _UnmarkedBenchmark(_LegacyBenchmark):
    identifier = 'unmarked-vision'
    subject_api = None  # not marked legacy: the benchmark receives the subject itself

    def __call__(self, candidate):
        output = neural_response(candidate, self.stimuli, ['neural:IT'], time_bins=[(70, 170)])
        score = Score([1.0], coords={'aggregation': ['center']}, dims=['aggregation'])
        score.attrs.update(candidate=type(candidate).__name__, output=output)
        return score


def test_score_wraps_a_legacy_benchmark_and_not_an_unmarked_one():
    model = BrainScoreModel('typed', region_layer_map={'IT': 'layer'},
                            extractor={'vision': _Extractor()}, visual_degrees=8)
    legacy = brainscore.score(model, _LegacyBenchmark(), check_mem=False)
    unmarked = brainscore.score(model, _UnmarkedBenchmark(), check_mem=False)
    assert legacy.attrs['candidate'] == 'LegacyBenchmarkFacade'
    assert legacy.attrs['degrees'] == 8
    assert unmarked.attrs['candidate'] == 'BrainScoreModel'
    np.testing.assert_array_equal(legacy.attrs['output'].values, unmarked.attrs['output'].values)
    assert legacy.attrs['compatibility']['match'] == 'full'


class _VisionNeuralBenchmark(_LegacyBenchmark):
    """What a legacy vision NeuralBenchmark does to its candidate."""

    def preallocate_memory(self, *args, **kwargs):
        pass


def test_typed_model_scores_identically_to_a_session(monkeypatch):
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
    session_output = neural_response(model, benchmark.stimuli, ['neural:IT'], time_bins=[(70, 170)])
    np.testing.assert_array_equal(score.attrs['output'].values, session_output.values)


# --- a refusal of an old benchmark's task: N/A before any input, an error after one ---

import pytest

from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import Subject


class _OldTaskBenchmark(_LegacyBenchmark):
    """An old behavior benchmark: it states no requested output and asks its task at run time."""
    identifier = 'legacy-odd-one-out'
    region = None

    def __call__(self, candidate):
        candidate.start_task('odd_one_out')
        candidate.look_at(self.stimuli)
        raise AssertionError('the subject refused the task, so the benchmark does not get here')


class _Refuser(Subject):
    """Refuses the task after it read ``inputs_before_refusal`` inputs."""
    identifier = 'refuser'
    in_channels = {'vision:image'}
    out_channels = {'behavior'}

    def __init__(self, inputs_before_refusal):
        self.inputs_before_refusal = inputs_before_refusal
        self.inputs_read = 0

    def participate(self, session):
        for _ in range(self.inputs_before_refusal):
            session.next_input()
            self.inputs_read += 1
        raise ExecutionNotFeasibleError(f"{self.identifier} has no answer for task_type "
                                        f"{session.task.task_type!r}", channels={'behavior'})


def test_an_old_task_refused_before_any_input_is_not_applicable():
    subject = _Refuser(inputs_before_refusal=0)
    result = brainscore.score(subject, _OldTaskBenchmark(), check_mem=False)
    assert result.attrs['status'] == 'N/A'
    assert 'no answer for task_type' in result.attrs['reason']
    assert result.attrs['missing_channels'] == ['behavior']
    assert np.isnan(result.values)
    assert subject.inputs_read == 0


def test_an_old_task_refused_after_an_input_stays_an_error():
    subject = _Refuser(inputs_before_refusal=1)
    with pytest.raises(ExecutionNotFeasibleError, match='no answer for task_type'):
        brainscore.score(subject, _OldTaskBenchmark(), check_mem=False)
    assert subject.inputs_read == 1
