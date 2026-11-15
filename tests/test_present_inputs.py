"""The inputs a task presents from a table: its ``present`` entries when it names them.

Notebook 14 shows image screens named through the task (``present=('image:instruction',
'image:word_1', ...)``) from columns that are not catalog columns. ``TaskProtocol``
declares the inputs with core's ``task_input_channels``, so such a protocol has the
input ``vision:image``, validates and runs, and pre-flight gives N/A to a subject that
takes none of its inputs. The table and tasks below are the notebook's forms, with file
paths in place of rendered images (no weights, no files read).
"""
import pytest

from brainscore import BrainScoreModel
from brainscore_core import Perturbation, StateChange, Subject
from brainscore_core.compatibility import CompatibilityError, check_channel_compatibility
from brainscore_core.contract import TaskContext
from brainscore_core.events import Selection
from brainscore_core.responses import Choice, Scale
from brainscore_core.streaming_helpers import task_input_channels
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.experiments import Experiment, Intervene, RecordBehavior, RecordNeural, TaskProtocol

pytestmark = pytest.mark.unit

CONFIDENCE_INSTRUCTION = ('Answer the question on the first screen for yourself, then rate how sure '
                          'you are of that answer: 1 means guessing, 5 means certain.')


def trial_table(length):
    """The notebook's trial table: one row per list and rule, screens as image paths."""
    rows = []
    for index in range(4):
        for rule in ('animal', 'colour'):
            row = dict(stimulus_id=f'L{length}-{index:02d}-{rule}', list_id=f'L{length}-{index:02d}',
                       rule=rule, length=length, filler='noun', animal_in=True, colour_in=False,
                       answer='yes' if rule == 'animal' else 'no', instruction=f'/screens/rule_{rule}.png')
            for position in range(1, length + 1):
                row[f'word_{position}'] = f'/screens/L{length}-{index:02d}_{position}.png'
                row[f'text_{position}'] = f'word{position}'
            rows.append(row)
    trials = StimulusSet(rows)
    trials.identifier = f'rsvp-reading-length{length}-test'
    return trials


def task(length, form='choice'):
    """The notebook's task card of one version: 'choice', 'confidence' or 'probabilities'."""
    screens = ('image:instruction',) + tuple(f'image:word_{i}' for i in range(1, length + 1))
    if form == 'confidence':
        return TaskContext(instruction=CONFIDENCE_INSTRUCTION, present=screens, response=Scale(1, 5, 1))
    return TaskContext(present=screens, response=Choice(['yes', 'no']))


def picture_subject(calls):
    def reply(prompt):
        calls.append([kind for kind, _ in prompt.parts])
        return '3' if isinstance(prompt.response, Scale) else 'yes'
    return BrainScoreModel('picture-reader', in_channels={'vision:image'}, reply=reply,
                           likelihood=lambda prompt, labels: {label: -float(i) for i, label in enumerate(labels)},
                           capability_config={'behavior-sessions': {
                               'behavior:probabilities': ('likelihood', 'softmax')}})


FORMS = {'choice': ([], 'choice'), 'confidence': ([], 'rating'),
         'probabilities': (['choice', 'probabilities'], 'probabilities')}


@pytest.mark.parametrize('length', [3, 6])
@pytest.mark.parametrize('form', sorted(FORMS))
def test_the_notebook_tables_validate_and_run(form, length, tmp_path):
    trials = trial_table(length)
    protocol = TaskProtocol(trials, task(length, form))
    assert protocol.required_input_channels == {'vision:image'}
    calls = []
    formats, measure = FORMS[form]
    result = Experiment(subject=picture_subject(calls), protocol=protocol,
                        tools=[RecordBehavior(formats or None)], output_dir=tmp_path / form).run()
    assert result.behavior(measure=measure).sizes['presentation'] == len(trials)
    if form != 'probabilities':
        # The reply saw the instruction screen and every word screen as images, then the
        # response's directive (after the text instruction for the confidence rating).
        images = ['image'] * (length + 1)
        assert calls[0] == (['text'] if form == 'confidence' else []) + images + ['text']


def test_a_text_only_subject_is_not_applicable():
    reader = BrainScoreModel('text-reader', in_channels={'text'}, reply=lambda prompt: 'yes')
    with pytest.raises(CompatibilityError, match='takes in none of the inputs'):
        check_channel_compatibility(reader, TaskProtocol(trial_table(3), task(3)))


def test_a_picture_only_subject_is_applicable():
    check_channel_compatibility(picture_subject([]), TaskProtocol(trial_table(3), task(3)))


def test_a_present_column_the_table_lacks_is_refused():
    with pytest.raises(ValueError, match="names column 'word_9', which the stimulus table does not have"):
        TaskProtocol(trial_table(3), TaskContext(present=('image:instruction', 'image:word_9'),
                                                 response=Choice(['yes', 'no'])))


def test_without_present_the_catalog_columns_are_the_inputs():
    table = StimulusSet([{'stimulus_id': 's0', 'sentence': 'a word', 'image_path': '/a.png', 'rule': 'x'}])
    assert task_input_channels(table) == {'text', 'vision:image'}
    assert task_input_channels(table, TaskContext(present=('text',), response=Choice(['a', 'b']))) == {'text'}


class _RecordingStandIn(Subject):
    """Declares what the notebook's recorded run asks for: image input, the two regions,
    an intervention and the choice. Used only for validation (no compute)."""
    identifier = 'qwen-stand-in'
    in_channels = {'vision:image', 'lesion'}
    out_channels = {'behavior:choice', 'behavior:rating', 'behavior:probabilities',
                    'neural:VWFA', 'neural:language_system', 'perturbation'}

    def participate(self, session):
        raise AssertionError('validation only')


@pytest.mark.parametrize('length', [3, 6])
def test_the_notebook_run_cell_builds_valid_experiments(length, tmp_path):
    """The run cell's three experiments, with a stand-in subject: each validates."""
    silence = StateChange(kind='ablation', target=Selection(layer='visual.blocks.20', indices=[0, 1]),
                          perturbation=Perturbation(kind='zero'))
    subject, trials = _RecordingStandIn(), trial_table(length)
    recorded = Experiment(
        subject=subject, protocol=TaskProtocol(trials, task(length), conditions=['intact', 'word_units']),
        tools=[RecordNeural(['VWFA', 'language_system']), RecordBehavior(),
               Intervene(silence, conditions=['word_units'], name='silence_word_units')],
        output_dir=tmp_path / 'run')
    rated = Experiment(subject=subject, protocol=TaskProtocol(trials, task(length, 'confidence')),
                       tools=[RecordBehavior()], output_dir=tmp_path / 'confidence')
    implied = Experiment(subject=subject, protocol=TaskProtocol(trials, task(length, 'probabilities')),
                         tools=[RecordBehavior(['choice', 'probabilities'])], output_dir=tmp_path / 'implied')
    for experiment in (recorded, rated, implied):
        experiment.validate()
