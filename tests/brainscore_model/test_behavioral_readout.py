"""Tests for BrainScoreModel behavioral readout.

Through behavioral_response (a session):
- a task with fitting stimuli gives a BehavioralAssembly of per-label probabilities
- the choice coordinate keeps the order in which the fitting labels first appear
- reset() discards the fitted readout; a passive task fits nothing
- a missing behavioral_readout_layer raises a clear error
"""


import numpy as np
import pandas as pd
import pytest

from brainscore import BrainScoreModel
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import behavioral_response, stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, NeuroidAssembly,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


# ── BrainScoreModel behavioral integration tests ───────────────

def _fake_preprocessor_returning(features_matrix):
    """An extractor that takes (stimuli, layers), returns a NeuroidAssembly,
    and keeps the stimulus ids of each call in ``calls``."""
    class FakeExtractor:
        identifier = 'fake-extractor'

        def __init__(self):
            self.calls = []

        def __call__(self, stimuli, layers=None, **kwargs):
            self.calls.append(list(stimuli['stimulus_id']))
            n = len(stimuli)
            # return features for the requested number of stimuli
            data = features_matrix[:n] if features_matrix.shape[0] >= n else \
                np.resize(features_matrix, (n, features_matrix.shape[1]))
            return NeuroidAssembly(
                data,
                coords={
                    'stimulus_id': ('presentation', list(stimuli['stimulus_id'].values)),
                    'neuroid_id': ('neuroid', [f'n{j}' for j in range(data.shape[1])]),
                },
                dims=['presentation', 'neuroid'],
            )
    return FakeExtractor()


def _make_image_stimulus_set(labels, identifier='test_stim'):
    n = len(labels)
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': [f's{i}' for i in range(n)],
        'image_file_name': [f'/tmp/s{i}.png' for i in range(n)],
        'image_label': labels,
    }))
    stimuli.identifier = identifier
    return stimuli


class TestBrainScoreModelBehavioral:
    @pytest.fixture
    def labeled_stimuli(self):
        return _make_image_stimulus_set(['cat'] * 6 + ['dog'] * 6)

    @pytest.fixture
    def scored_stimuli(self):
        return _make_image_stimulus_set(['cat'] * 3 + ['dog'] * 3, identifier='test_stim2')

    def test_behavior_session_returns_behavioral_assembly(self, labeled_stimuli, scored_stimuli):
        features = np.random.default_rng(42).normal(size=(12, 8))
        # Make labels linearly separable for stability
        features[:6, 0] += 3.0  # cat rows shift
        features[6:, 0] -= 3.0  # dog rows shift
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            extractor={'vision': extractor},
            behavioral_readout_layer='some_layer',
        )
        task = TaskContext(response=Choice(['cat', 'dog']), fitting_stimuli=labeled_stimuli)
        session = stimulus_session(scored_stimuli, ['behavior:probabilities'], task)
        model.participate(session)
        result = session.collect('behavior:probabilities')
        assert isinstance(result, BehavioralAssembly)
        assert result.dims == ('presentation', 'choice')
        assert result.shape == (6, 2)
        # Sums to 1
        assert np.allclose(result.values.sum(axis=1), 1.0, atol=1e-6)
        assert result.attrs['response']['method'] == 'readout'


    def test_reset_clears_classifier(self, labeled_stimuli, scored_stimuli):
        features = np.random.default_rng(0).normal(size=(12, 8))
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            extractor={'vision': extractor},
            behavioral_readout_layer='some_layer',
            answer_owners={'behavior:choice': ('readout', 'argmax')},
        )
        task = TaskContext(response=Choice(['cat', 'dog']), fitting_stimuli=labeled_stimuli)
        fitting_ids = list(labeled_stimuli['stimulus_id'])

        def fits():
            return extractor.calls.count(fitting_ids)
        behavioral_response(model, scored_stimuli, task)
        behavioral_response(model, scored_stimuli, task)   # the same task keeps its fitted readout
        assert fits() == 1
        model.reset()
        behavioral_response(model, scored_stimuli, task)   # after reset the readout is fitted again
        assert fits() == 2


    def test_passive_task_does_not_fit(self, labeled_stimuli):
        features = np.random.default_rng(0).normal(size=(12, 8))
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={'IT': 'some_layer'},
            extractor={'vision': extractor},
            behavioral_readout_layer='some_layer',
        )
        # 'passive' is neural-style, no readout needed: the one extraction is the recording
        model.participate(stimulus_session(labeled_stimuli, ['neural:IT'], TaskContext(task_type='passive')))
        assert len(extractor.calls) == 1


class TestBrainScoreModelGenerationPath:
    """Validate the alternative instruction+generation dispatch path."""

    @pytest.fixture
    def scored_stimuli(self):
        return _make_image_stimulus_set(
            ['cat'] * 3 + ['dog'] * 3, identifier='gen_test')

    def test_generation_path_routes_correctly(self, scored_stimuli):
        """Model with an answer function and TaskContext(instruction, label_set)
        should dispatch to generation, not readout."""
        calls = []

        def fake_generate(stimulus_row, instruction, label_set):
            calls.append((stimulus_row['stimulus_id'], instruction, tuple(label_set)))
            # Pretend the generation returns the stimulus label (e.g., VLM
            # reads 'cat' vs 'dog' correctly for even-indexed stimuli).
            sid = stimulus_row['stimulus_id']
            return 'cat' if sid in {'s0', 's1', 's2'} else 'dog'

        model = BrainScoreModel(
            identifier='test-vlm',
            model=None,
            region_layer_map={},
            extractor={'vision': lambda *a, **kw: None},
            answer=fake_generate,
        )
        task = TaskContext(response=Choice(['cat', 'dog']), instruction='What animal is this?')
        result = behavioral_response(model, scored_stimuli, task)
        assert result.dims == ('presentation', 'choice')
        assert result.shape == (6, 2)
        # Each row should be a one-hot
        assert np.allclose(result.values.sum(axis=1), 1.0)
        # the answer function should be called once per stimulus
        assert len(calls) == 6
        # Instruction should be forwarded verbatim
        assert all(c[1] == 'What animal is this?' for c in calls)



def test_a_readout_layer_whose_features_keep_a_time_axis_is_refused_with_a_clear_error():
    """A readout fits one feature vector per stimulus. Features with a time axis are refused
    with an error that names the layer and the time axis; nothing reduces over time silently."""
    def extract(stimuli, layers=None, **kwargs):
        ids = list(stimuli['stimulus_id'])
        return NeuroidAssembly(np.zeros((len(ids), 2, 3)), dims=['presentation', 'neuroid', 'time_bin'], coords={
            'stimulus_id': ('presentation', ids), 'image_label': ('presentation', list(stimuli['image_label'])),
            'neuroid_id': ('neuroid', ['n0', 'n1']), 'layer': ('neuroid', ['clip_layer'] * 2),
            'time_bin_start_ms': ('time_bin', [0, 100, 200]), 'time_bin_end_ms': ('time_bin', [100, 200, 300])})
    model = BrainScoreModel('time-axis-readout', extractor={'vision': extract},
                            behavioral_readout_layer='clip_layer', answer_owners={'behavior:choice': ('readout', 'argmax')},)
    task = TaskContext(response=Choice(['cat', 'dog']),
                       fitting_stimuli=_make_image_stimulus_set(['cat', 'dog'] * 3, identifier='time_fit'))
    with pytest.raises(ExecutionNotFeasibleError, match=r"'clip_layer' keep a time axis"):
        behavioral_response(model, _make_image_stimulus_set(['cat', 'dog'], identifier='time_test'), task)
