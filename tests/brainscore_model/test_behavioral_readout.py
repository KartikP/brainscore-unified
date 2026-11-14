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
from brainscore_core.compatibility import CompatibilityError
from brainscore_core.contract import TaskContext
from brainscore_core.streaming_helpers import behavioral_response
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
        task = TaskContext(task_type='probabilities', fitting_stimuli=labeled_stimuli)
        result = behavioral_response(model, scored_stimuli, task)
        assert isinstance(result, BehavioralAssembly)
        assert result.dims == ('presentation', 'choice')
        assert result.shape == (6, 2)
        # Sums to 1
        assert np.allclose(result.values.sum(axis=1), 1.0, atol=1e-6)
        assert result.attrs['measure'] == {'kind': 'implied', 'source': 'readout'}

    def test_choice_keeps_the_first_seen_order_of_the_fitting_labels(self):
        labels = ['z', 'a', 'z', 'm', 'a', 'z', 'm', 'a', 'z']
        fitting = _make_image_stimulus_set(labels, identifier='order_fit')
        features = np.random.default_rng(0).normal(size=(9, 8))
        model = BrainScoreModel('test-model', extractor={'vision': _fake_preprocessor_returning(features)},
                                behavioral_readout_layer='some_layer')
        result = behavioral_response(model, _make_image_stimulus_set(['z', 'a'], identifier='order_test'),
                                     TaskContext(task_type='probabilities', fitting_stimuli=fitting))
        assert list(result['choice'].values) == ['z', 'a', 'm']

    def test_process_returns_neural_before_task(self, labeled_stimuli):
        features = np.random.default_rng(0).normal(size=(12, 8))
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            extractor={'vision': extractor},
            behavioral_readout_layer='some_layer',
        )
        # Without start_task, process() returns neural features
        result = model.process(labeled_stimuli)
        assert isinstance(result, NeuroidAssembly)
        assert 'neuroid' in result.dims

    def test_reset_clears_classifier(self, labeled_stimuli):
        features = np.random.default_rng(0).normal(size=(12, 8))
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            extractor={'vision': extractor},
            behavioral_readout_layer='some_layer',
        )
        task = TaskContext(task_type='probabilities', fitting_stimuli=labeled_stimuli)
        model.start_task(task)
        model.start_task(task)                   # the same task keeps its fitted readout
        assert len(extractor.calls) == 1
        model.reset()
        model.start_task(task)                   # after reset the readout is fitted again
        assert len(extractor.calls) == 2

    def test_missing_readout_layer_raises(self, labeled_stimuli):
        features = np.random.default_rng(0).normal(size=(12, 8))
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            extractor={'vision': extractor},
            # No behavioral_readout_layer
        )
        with pytest.raises(CompatibilityError, match="behavioral_readout_layer"):
            model.start_task(TaskContext(
                task_type='probabilities',
                fitting_stimuli=labeled_stimuli,
            ))

    def test_passive_task_does_not_fit(self, labeled_stimuli):
        features = np.random.default_rng(0).normal(size=(12, 8))
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            extractor={'vision': extractor},
            behavioral_readout_layer='some_layer',
        )
        # 'passive' is neural-style, no readout needed
        model.start_task(TaskContext(task_type='passive'))
        assert extractor.calls == []             # nothing was fitted


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
        task = TaskContext(
            task_type='probabilities',
            label_set=['cat', 'dog'],
            instruction='What animal is this?',
        )
        result = behavioral_response(model, scored_stimuli, task)
        assert result.dims == ('presentation', 'choice')
        assert result.shape == (6, 2)
        # Each row should be a one-hot
        assert np.allclose(result.values.sum(axis=1), 1.0)
        # the answer function should be called once per stimulus
        assert len(calls) == 6
        # Instruction should be forwarded verbatim
        assert all(c[1] == 'What animal is this?' for c in calls)

    def test_readout_path_still_works_when_no_instruction(self, scored_stimuli):
        """If TaskContext has fitting_stimuli but no instruction, dispatch
        to the readout path even when the model has an answer function."""
        features = np.random.default_rng(0).normal(size=(6, 8))
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            extractor={'vision': extractor},
            behavioral_readout_layer='some_layer',
            answer=lambda **kw: 'cat',  # shouldn't be called
        )
        fitting = _make_image_stimulus_set(['cat'] * 3 + ['dog'] * 3,
                                            identifier='gen_fit')
        result = behavioral_response(model, scored_stimuli, TaskContext(
            task_type='probabilities',
            fitting_stimuli=fitting,
            label_set=['cat', 'dog'],
            # no instruction
        ))
        # The answer part takes keywords only, so a call would raise.
        assert result.attrs['measure'] == {'kind': 'implied', 'source': 'readout'}

    def test_error_when_no_path_viable(self, scored_stimuli):
        """TaskContext with no fitting_stimuli, no answer function, no
        instruction+label_set combo → clear error."""
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            extractor={'vision': lambda *a, **kw: None},
        )
        with pytest.raises(CompatibilityError, match="Paths tried: generation needs answer=.*readout needs TaskContext.fitting_stimuli"):
            model.start_task(TaskContext(
                task_type='probabilities',
                label_set=['cat', 'dog'],
                # no fitting_stimuli and no answer function
            ))
