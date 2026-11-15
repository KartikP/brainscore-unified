"""Old models wrapped by the permanent adapters stay applicable on ROAR (review S7-01).

ROAR requests ``behavior:choice``. The vision and language adapters answer a Choice task
from the old model's own forced-choice task (``'probabilities'``), as the old ROAR asked
it on bare ``behavior``, and turn its rows into the choice by core's ``argmax``. The
expected score 0.51 (real 0.68, pseudo 0.34) is what the same stand-ins scored through
``brainscore.score`` on the code before ROAR's switch (unified fe1a799, core 92f85d9,
vision 9317ca34, language 6163650), measured once on that tree.
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
    """A legacy language subject: start_behavioral_task, digest_text -> {'behavior': rows}."""

    def __init__(self):
        self.tasks = []

    def identifier(self):
        return 'legacy-language-stand-in'

    def start_behavioral_task(self, task):
        self.tasks.append(task)

    def digest_text(self, text):
        return {'behavior': _rows([f's{int(word.split("-")[1])}' for word in text])}


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


def _adapters():
    from brainscore_language.compat.unified_adapter import LanguageModelAdapter
    from brainscore_vision.compat.unified_adapter import VisionModelAdapter
    return {'vision': (lambda: VisionModelAdapter(LegacyVisionModel()), 'vision:image'),
            'language': (lambda: LanguageModelAdapter(LegacyLanguageModel()), 'text')}


@pytest.mark.parametrize('kind', ['vision', 'language'])
def test_an_old_model_scores_roar_as_before(kind, roar_words):
    build, channel = _adapters()[kind]
    subject = build()
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


def test_an_old_model_with_no_behavior_task_stays_not_applicable(roar_words):
    from brainscore_language.compat.unified_adapter import LanguageModelAdapter
    subject = LanguageModelAdapter(LegacyLanguageModel(), out_channels={'neural:language_system'})
    score = brainscore.score(subject, Yeatman2021LexicalDecision(channels=('text',)), check_mem=False)
    assert score.attrs['status'] == 'N/A' and subject._legacy.tasks == []
