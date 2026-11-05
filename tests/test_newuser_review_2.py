"""The second new-user review: full recording with every answer form, reopening a run,
trial tables with stimulus columns and answer details, and GridGame compatibility."""
import pytest

from brainscore_core import BrainScoreModel, TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.experiments import (Experiment, ExperimentResult, RecordBehavior,
                                    RecordInputsOutputs, TaskProtocol)

pytestmark = pytest.mark.unit

STIMULI = StimulusSet({'stimulus_id': ['cat', 'boat', 'unknown'],
                       'sentence': ['cat', 'boat', 'ambiguous'], 'word_length': [3, 4, 9]})
STIMULI.identifier = 'new-user-text-trials'


def _run(response, replies, directory):
    def reply(prompt, _seen=[]):
        _seen.append(prompt)
        return replies[(len(_seen) - 1) % len(replies)]
    subject = BrainScoreModel(identifier='scripted', supported_modalities={'text'}, reply=reply)
    task = TaskContext(instruction='Respond to the word.', response=response, prefer_path='generation')
    return Experiment(subject=subject,
                      protocol=TaskProtocol(STIMULI, task, conditions=['baseline'], repetitions=[0]),
                      tools=[RecordBehavior(), RecordInputsOutputs()], output_dir=directory).run()


@pytest.mark.parametrize('response, replies, column', [
    (Choice(['animal', 'vehicle']), ['animal', 'vehicle', 'uncertain'], 'choice'),
    (Scale(1, 7, 1), ['6', '3', '9'], 'rating'),
    (Text(), ['A cat.', 'A boat.', 'I cannot tell.'], 'text')])
def test_full_recording_works_with_every_answer_form(tmp_path, response, replies, column):
    result = _run(response, replies, tmp_path / 'run')
    assert len(list(result.record.outputs())) > 0           # the input/output record was written
    trials = ExperimentResult.open(tmp_path / 'run').trials()
    assert list(trials['word_length']) == [3, 4, 9]          # stimulus columns come along
    assert column in trials and f'{column}_valid' in trials


def test_invalid_answers_keep_the_reply_and_the_reason(tmp_path):
    trials = _run(Choice(['animal', 'vehicle']), ['animal', 'vehicle', 'uncertain'], tmp_path / 'r').trials()
    unknown = trials.set_index('stimulus_id').loc['unknown']
    assert unknown['choice'] is None and not unknown['choice_valid']
    assert unknown['choice_raw'] == 'uncertain' and unknown['choice_error']


def test_open_refuses_a_folder_that_is_not_a_run(tmp_path):
    with pytest.raises(FileNotFoundError, match='not a saved experiment run'):
        ExperimentResult.open(tmp_path)


def test_grid_game_is_not_applicable_to_a_subject_without_a_policy():
    import brainscore
    from brainscore_core.compatibility import CompatibilityError, check_compatibility
    from brainscore.benchmarks.grid_game.benchmark import GridGameBenchmark, policy_model
    from brainscore.harnesses.grid_game import greedy_oracle_policy
    benchmark = GridGameBenchmark(size=5, n_episodes=1)
    with pytest.raises(CompatibilityError, match='motor'):
        check_compatibility(brainscore.load_model('chance-baseline'), benchmark)
    check_compatibility(policy_model(greedy_oracle_policy, 'oracle'), benchmark)
