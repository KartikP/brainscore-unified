"""Old models wrapped by the permanent adapters on ROAR, which requests behavior:choice.

Vision (review S7-01): the adapter answers a Choice task from the old model's own
forced-choice task (``'probabilities'`` with fitting stimuli, which a ModelCommitment
has), as the old ROAR asked it on bare ``behavior``, and turns its rows into the choice by
core's ``argmax``. The expected score 0.51 (real 0.68, pseudo 0.34) is what the same
stand-in scored through ``brainscore.score`` on the code before ROAR's switch (unified
fe1a799, core 92f85d9, vision 9317ca34), measured once on that tree.

Language: an old language model has only the tasks next_word and reading_times, so the
adapter declares no choice and the pair is N/A before any call (a stated change: before,
the pair was applicable by declaration and failed during the run).
"""
import numpy as np
import pytest

import brainscore
from brainscore.benchmarks.roar_yeatman2021.benchmark import Yeatman2021LexicalDecision
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


def _p_real(stimulus_id):
    number = int(str(stimulus_id).lstrip('s'))
    return 0.8 if number % 4 else 0.3        # every fourth word called pseudo


def _rows(stimulus_ids):
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.array([[1 - _p_real(sid), _p_real(sid)] for sid in stimulus_ids])
    return BehavioralAssembly(values, dims=['presentation', 'choice'], coords={
        'stimulus_id': ('presentation', list(stimulus_ids)),
        'position': ('presentation', list(range(len(stimulus_ids)))), 'choice': ['pseudo', 'real']})


class LegacyVisionModel:
    """A legacy vision BrainModel: start_task('probabilities', fitting), look_at -> rows."""
    identifier = 'legacy-vision-stand-in'

    def __init__(self):
        self.tasks = []

    def start_task(self, task, fitting_stimuli=None, **kwargs):
        self.tasks.append(task)

    def look_at(self, stimuli, number_of_trials=1):
        return _rows(stimuli['stimulus_id'])

    def visual_degrees(self):
        return 8


class LegacyLanguageModel:
    """An old language subject with the real behavioral tasks: next_word and reading_times."""

    def __init__(self):
        self.tasks = []

    def identifier(self):
        return 'legacy-language-stand-in'

    def start_behavioral_task(self, task):
        if task not in ('next_word', 'reading_times'):
            raise NotImplementedError(f'task {task!r} is not supported')
        self.tasks.append(task)

    def digest_text(self, text):
        raise AssertionError('never called on ROAR')


@pytest.fixture
def roar_words(monkeypatch):
    from brainscore.benchmarks.roar_yeatman2021 import benchmark as roar
    ids = [f's{i}' for i in range(500)]
    labels = ['real'] * 250 + ['pseudo'] * 250
    stimuli = StimulusSet({'stimulus_id': ids, 'sentence': [f'{l}-{i}' for i, l in enumerate(labels)],
                           'image_file_name': [f'/x/{i}.png' for i in ids], 'image_label': labels})
    stimuli.identifier = 'synthetic-roar-legacy'
    stimuli.stimulus_paths = {i: f'/x/{i}.png' for i in ids}
    monkeypatch.setattr(roar, 'load_stimulus_set', lambda identifier: stimuli)
    monkeypatch.setattr(roar, 'load_dataset', lambda identifier: None)
    monkeypatch.setattr(roar, '_human_accuracy_on_stimuli', lambda *a, **k: 1.0)


def test_an_old_vision_model_scores_roar_image_as_before(roar_words):
    from brainscore_vision.compat.unified_adapter import VisionModelAdapter
    subject, channel = VisionModelAdapter(LegacyVisionModel()), 'vision:image'
    assert {'behavior', 'behavior:choice'} <= subject.out_channels
    sessions = []
    participate = subject.participate
    subject.participate = lambda session: sessions.append(session) or participate(session)
    score = brainscore.score(subject, Yeatman2021LexicalDecision(channels=(channel,)), check_mem=False)
    assert 'status' not in score.attrs
    assert (float(score.attrs['raw']), score.attrs['accuracy_real'], score.attrs['accuracy_pseudo']) == (
        0.51, 0.68, 0.34)
    assert subject._legacy.tasks == ['probabilities']
    record = sessions[0].collect('behavior:choice').attrs['response']
    assert (record['origin'], record['method'], record['conversion'], record['source_format']) == (
        'derived', 'legacy', 'argmax', 'behavior:probabilities')


def test_an_old_language_model_is_not_applicable_on_roar_text_before_any_call(roar_words):
    from brainscore_language.compat.unified_adapter import LanguageModelAdapter
    legacy = LegacyLanguageModel()
    subject = LanguageModelAdapter(legacy)
    assert 'behavior:choice' not in subject.out_channels
    score = brainscore.score(subject, Yeatman2021LexicalDecision(channels=('text',)), check_mem=False)
    assert score.attrs['status'] == 'N/A' and 'behavior:choice' in score.attrs['reason']
    assert legacy.tasks == []


def test_an_old_model_with_no_behavior_task_stays_not_applicable(roar_words):
    from brainscore_language.compat.unified_adapter import LanguageModelAdapter
    subject = LanguageModelAdapter(LegacyLanguageModel(), out_channels={'neural:language_system'})
    score = brainscore.score(subject, Yeatman2021LexicalDecision(channels=('text',)), check_mem=False)
    assert score.attrs['status'] == 'N/A' and subject._legacy.tasks == []


# Moved from vision tests/test_unified_adapter.py (S7-05): the score path through
# brainscore.score belongs here, because the vision tests run without the unified package.

class _FileFeatures:
    """An offline extractor: one feature per image, read from the image file itself."""

    def __init__(self):
        self.shown = []

    def __call__(self, stimuli, layers, number_of_trials=1, require_variance=False):
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
        values = []
        for stimulus_id in stimuli['stimulus_id']:
            with open(stimuli.get_stimulus(stimulus_id), 'rb') as image:
                values.append([float(len(image.read()))])
            self.shown.append(stimulus_id)
        return NeuroidAssembly(np.array(values), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
            'image_label': ('presentation', list(stimuli['image_label'])),
            'neuroid_id': ('neuroid', [0]), 'layer': ('neuroid', ['L'])})


def _image_set(folder, prefix, missing=()):
    ids = [f'{prefix}{i}' for i in range(4)]
    paths = {sid: str(folder / f'{sid}.png') for sid in ids}
    for sid, path in paths.items():
        if sid not in missing:
            with open(path, 'wb') as image:
                image.write(b'x' * (10 + ids.index(sid)))
    stimuli = StimulusSet({'stimulus_id': ids, 'image_file_name': list(paths.values()),
                           'image_label': ['real', 'pseudo'] * 2})
    stimuli.identifier = f'adapter-choice-{prefix}'
    stimuli.stimulus_paths = paths
    return stimuli


class _ChoiceBenchmark:
    """A benchmark that requests behavior:choice with fitting stimuli, as ROAR does."""
    identifier = 'adapter-choice-benchmark'
    required_input_channels = frozenset({'vision:image'})
    requested_output_channels = frozenset({'behavior:choice'})
    trial_memory = 'none'

    def __init__(self, test, fitting):
        self.test, self.fitting, self.sessions = test, fitting, []

    def __call__(self, subject):
        from brainscore_core.contract import TaskContext
        from brainscore_core.responses import Choice
        from brainscore_core.streaming_helpers import stimulus_session
        session = stimulus_session(self.test, ['behavior:choice'], TaskContext(
            response=Choice(['real', 'pseudo']), fitting_stimuli=self.fitting))
        self.sessions.append(session)
        subject.participate(session)
        raise AssertionError('the set-up error did not stop the run')


class _LegacyWithoutProbabilities(LegacyVisionModel):
    """A legacy BrainModel that is not a ModelCommitment and has no probabilities task."""
    identifier = 'legacy-label-only'

    def start_task(self, task, fitting_stimuli=None, **kwargs):
        if task != 'label':
            raise NotImplementedError(f'unsupported task {task}')
        self.tasks.append(task)


def test_a_missing_fitting_file_passes_through_as_file_not_found(tmp_path):
    """An operational error of the old model's readout fit is not turned into a refusal."""
    from brainscore_vision.compat.unified_adapter import VisionModelAdapter
    from brainscore_vision.model_helpers.brain_transformation.behavior import ProbabilitiesMapping

    class _Mapping(ProbabilitiesMapping):
        def visual_degrees(self):
            return 8

    features = _FileFeatures()
    legacy = _Mapping('legacy-probabilities', features, 'L')
    benchmark = _ChoiceBenchmark(_image_set(tmp_path, 't'), _image_set(tmp_path, 'f', missing={'f2'}))
    with pytest.raises(FileNotFoundError, match='f2.png'):
        brainscore.score(VisionModelAdapter(legacy), benchmark, check_mem=False)
    assert features.shown == ['f0', 'f1']                  # no test stimulus is shown
    assert benchmark.sessions[0].emitted == []             # nothing is stored


def test_a_legacy_model_that_refuses_probabilities_raises_its_own_error(tmp_path):
    from brainscore_vision.compat.unified_adapter import VisionModelAdapter
    legacy = _LegacyWithoutProbabilities()
    benchmark = _ChoiceBenchmark(_image_set(tmp_path, 't'), _image_set(tmp_path, 'f'))
    with pytest.raises(NotImplementedError, match='unsupported task probabilities'):
        brainscore.score(VisionModelAdapter(legacy), benchmark, check_mem=False)
    assert legacy.tasks == [] and benchmark.sessions[0].emitted == []
