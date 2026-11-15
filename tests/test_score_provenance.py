"""The provenance of a run's responses reaches the score (rule 6.3), and moves no value."""
import numpy as np
import pytest

import brainscore
from brainscore import BrainScoreModel
from brainscore.benchmarks import benchmark_score, not_estimated
from brainscore.model_helpers.provenance import ProvenanceSubject
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.contract import TaskContext
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from tests.test_wrapper_cache_guard import _stimuli, text_wrapper

pytestmark = pytest.mark.integration


class _MeanResponse(BenchmarkBase):
    """The mean recorded IT response to three sentences."""
    requested_output_channels = frozenset({'neural:IT'})

    def __init__(self):
        self.stimulus_set = _stimuli('provenance-stimuli', sentence=['the cat sat on the mat', 'dog ran',
                                                                     'the dog sat'])
        super().__init__(identifier='provenance-fixture', version=1, parent='neural',
                         ceiling=not_estimated())

    def __call__(self, subject):
        session = stimulus_session(self.stimulus_set, ['neural:IT'], TaskContext(task_type='passive'))
        subject.participate(session)
        (event,) = session.emitted
        return benchmark_score(self, raw=float(event.payload.values.mean()), metric='mean')


def _model(**settings):
    return BrainScoreModel('provenance-text', extractor={'text': text_wrapper(**settings)},
                           region_layer_map={'IT': 'proj'})


def _score(subject):
    return brainscore.score(subject, _MeanResponse(), check_mem=False)


@pytest.fixture(autouse=True)
def _store_off(tmp_path, monkeypatch):
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path / 'store'))


def test_the_extraction_fingerprint_names_the_batch_size_and_moves_no_value():
    two, three = _score(_model(batch_size=2)), _score(_model(batch_size=3))
    (first,), (second,) = two.attrs['provenance'], three.attrs['provenance']
    assert first['extractor'] == 'brainscore.model_helpers.text_wrapper.TextWrapper'
    assert first['extraction'].startswith('v2-') and first['activation_store'] == 'off'
    assert first['extraction'] != second['extraction']
    assert float(two) == float(three)


def test_the_store_on_and_off_give_one_fingerprint_and_one_value(monkeypatch):
    off = _score(_model(batch_size=2))
    monkeypatch.delenv('RESULTCACHING_DISABLE')
    cold, warm = _score(_model(batch_size=2)), _score(_model(batch_size=2))
    for on in (cold, warm):
        (stamped,) = on.attrs['provenance']
        assert stamped['activation_store'] == 'on'
        assert stamped['extraction'] == off.attrs['provenance'][0]['extraction']
        assert float(on) == float(off)


def test_the_model_provenance_of_a_decorated_subject_reaches_the_score():
    model = {'dtype': 'torch.float32', 'revision': 'fixture'}
    score = _score(ProvenanceSubject(_model(batch_size=2), model))
    assert model in score.attrs['provenance']
    assert float(score) == float(_score(_model(batch_size=2)))


def test_an_extractor_without_provenance_gives_an_empty_tuple():
    def extract(stimuli, layers):
        return NeuroidAssembly(np.ones((len(stimuli), 2)), dims=['presentation', 'neuroid'],
                               coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                                       'layer': ('neuroid', ['proj', 'proj'])})
    subject = BrainScoreModel('plain', extractor={'text': extract}, region_layer_map={'IT': 'proj'})
    score = _score(subject)
    assert score.attrs['provenance'] == ()
    assert float(score) == 1.0
