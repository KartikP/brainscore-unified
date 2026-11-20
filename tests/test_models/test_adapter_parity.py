"""The adapters follow the core rules: a whole-table answer has the time the core driver gives
it (none, rule 4.2), and a second session gives what a fresh subject gives."""
import numpy as np
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.streaming_helpers import (
    behavioral_response, drive_neural_session, neural_response, stimulus_session)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, NeuroidAssembly)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


def _table(column):
    stimuli = StimulusSet({'stimulus_id': ['s0', 's1'], column: ['the cat', 'sat on'],
                           't_ms': [10.0, 20.0]})
    stimuli.identifier = f'adapter-parity-{column}'
    return stimuli


class _OldVisionModel:
    identifier = 'old-vision'

    class layer_model:
        region_layer_map = {'IT': 'layer4'}

    def visual_degrees(self):
        return 8

    def start_recording(self, region, time_bins=None):
        pass

    def start_task(self, task, fitting_stimuli=None):
        pass

    def look_at(self, stimuli, number_of_trials=1):
        ids = list(stimuli['stimulus_id'])
        return NeuroidAssembly(np.zeros((len(ids), 2)), dims=['presentation', 'neuroid'],
                               coords={'stimulus_id': ('presentation', ids),
                                       'neuroid_id': ('neuroid', ['a', 'b'])})


class _OldLanguageModel:
    """Keeps its task between calls; with a task set it reads each text as a prompt,
    so its activations differ (1 instead of 0)."""

    def __init__(self):
        self.region_layer_mapping = {'language_system': 'transformer.h.5'}
        self.neural_recordings, self.behavioral_task = [], None

    def identifier(self):
        return 'old-language'

    def start_neural_recording(self, target, recording_type):
        self.neural_recordings.append((target, recording_type))

    def start_behavioral_task(self, task):
        self.behavioral_task = task

    def digest_text(self, text):
        text = [text] if isinstance(text, str) else list(text)
        coords = {'stimulus': ('presentation', text),
                  'part_number': ('presentation', list(range(len(text))))}
        out = {}
        if self.behavioral_task is not None:
            out['behavior'] = BehavioralAssembly(np.array(['w'] * len(text)), coords=coords,
                                                 dims=['presentation'])
        if self.neural_recordings:
            value = float(self.behavioral_task is not None)
            out['neural'] = NeuroidAssembly(np.full((len(text), 2), value),
                                            dims=['presentation', 'neuroid'],
                                            coords={**coords, 'neuroid_id': ('neuroid', ['a', 'b'])})
        return out


def _adapters():
    """(a fresh adapter, a table it reads, a region it records) for each adapter."""
    from brainscore_vision.compat.unified_adapter import VisionModelAdapter
    from brainscore_language.compat.unified_adapter import LanguageModelAdapter
    return {
        'vision': (lambda: VisionModelAdapter(_OldVisionModel()), _table('image_path'), 'IT'),
        'language': (lambda: LanguageModelAdapter(_OldLanguageModel()), _table('sentence'),
                     'language_system'),
    }


@pytest.mark.parametrize('name', sorted(_adapters()))
def test_a_whole_table_answer_has_the_time_the_core_driver_gives(name):
    make, stimuli, region = _adapters()[name]

    def times(run):
        session = stimulus_session(stimuli, [f'neural:{region}'], TaskContext(task_type='passive'))
        run(make(), session)
        return [event.t_ms for event in session.emitted]

    assert times(lambda subject, session: subject.participate(session)) == times(drive_neural_session)


def _second_session_gives_what_a_fresh_subject_gives(make, first, second):
    used = make()
    first(used)
    return second(used), second(make())


def test_a_second_session_on_an_adapter_is_a_fresh_session():
    """A passive session after a task session gives what it gives on a fresh adapter."""
    make, stimuli, region = _adapters()['language']
    after_a_task, fresh = _second_session_gives_what_a_fresh_subject_gives(
        make,
        lambda subject: behavioral_response(subject, stimuli, TaskContext(task_type='next_word')),
        lambda subject: neural_response(subject, stimuli, [f'neural:{region}']).values.tolist())
    assert after_a_task == fresh
