"""chance-baseline as the floor of old benchmarks that run through the legacy facade."""
import numpy as np
import pandas as pd
import pytest

from brainscore.models.chance_baseline.model import ChanceBaseline
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.legacy_facade import LegacyBenchmarkFacade
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _stimuli(ids, labels=None, identifier='chance-old'):
    columns = {'stimulus_id': list(ids), 'object_name': ['o'] * len(ids)}
    if labels is not None:
        columns['image_label'] = list(labels)
    table = StimulusSet(pd.DataFrame(columns))
    table.identifier = identifier
    return table


def test_an_old_probabilities_task_with_fitting_stimuli_is_answered_uniformly():
    fitting = _stimuli(['f1', 'f2', 'f3', 'f4'], ['dog', 'cat', 'dog', 'bird'], 'chance-fit')
    test = _stimuli(['t1', 't2', 't3'], identifier='chance-test')
    facade = LegacyBenchmarkFacade(ChanceBaseline(), visual_degrees=8)
    facade.start_task('probabilities', fitting_stimuli=fitting)
    answer = facade.look_at(test)
    assert answer.dims == ('presentation', 'choice')
    assert list(answer['choice'].values) == ['dog', 'cat', 'bird']          # the readout's order: first occurrence
    np.testing.assert_allclose(answer.values, 1 / 3)
    assert list(answer['stimulus_id'].values) == ['t1', 't2', 't3']          # every stimulus answered
    assert 'object_name' in answer.indexes['presentation'].names             # the stimulus columns are there


def test_the_label_order_is_the_order_of_the_readout_classifier():
    from brainscore._model.fitted_readout import ProbabilitiesClassifier
    from brainscore._model.behavioral import BehavioralReadout
    fitting = _stimuli(['a', 'b', 'c', 'd', 'e'], ['z', 'y', 'z', 'x', 'y'], 'chance-order')
    classifier = ProbabilitiesClassifier()
    classifier.fit(np.random.RandomState(0).randn(5, 3), BehavioralReadout.extract_labels(fitting))
    facade = LegacyBenchmarkFacade(ChanceBaseline(), visual_degrees=8)
    facade.start_task('probabilities', fitting_stimuli=fitting)
    assert list(facade.look_at(_stimuli(['t'], identifier='chance-order-t'))['choice'].values) == classifier.label_set


def test_a_task_with_a_label_set_is_answered_as_before():
    model = ChanceBaseline()
    model.start_task(TaskContext(task_type='probabilities', label_set=['b', 'a']))
    answer = model.process(_stimuli(['t1', 't2']))
    assert list(answer['choice'].values) == ['b', 'a']
    np.testing.assert_allclose(answer.values, 0.5)


@pytest.mark.parametrize('task', ['odd_one_out', 'reading_times'])
def test_a_task_without_a_chance_answer_is_refused_by_name(task):
    facade = LegacyBenchmarkFacade(ChanceBaseline(), visual_degrees=8)
    facade.start_task(task)
    with pytest.raises(ExecutionNotFeasibleError, match=task):
        facade.look_at(_stimuli(['t1', 't2', 't3'], identifier='chance-refused'))


def test_a_probabilities_task_with_no_labels_anywhere_is_refused():
    facade = LegacyBenchmarkFacade(ChanceBaseline(), visual_degrees=8)
    with pytest.raises(ValueError, match='labels'):
        facade.start_task('probabilities', fitting_stimuli=_stimuli(['f1', 'f2'], identifier='chance-nolabels'))


def test_a_state_change_is_refused_with_what_to_ask_for():
    from brainscore_core.events import StateChange
    with pytest.raises(ExecutionNotFeasibleError, match='behavior'):
        ChanceBaseline().process(StateChange(kind='lesion', target='IT'))


def test_a_request_for_trial_to_trial_variance_is_refused():
    """Identical rows are no variance (Contract Rules 6.2): the request is refused, not answered."""
    fitting = _stimuli(['f1', 'f2'], ['dog', 'cat'], 'chance-variance-fit')
    facade = LegacyBenchmarkFacade(ChanceBaseline(), visual_degrees=8)
    facade.start_task('probabilities', fitting_stimuli=fitting)
    with pytest.raises(ExecutionNotFeasibleError, match='trial-to-trial variance'):
        facade.look_at(_stimuli(['t1', 't2'], identifier='chance-variance'), number_of_trials=3,
                       require_variance=True)
