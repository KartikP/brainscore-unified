"""score() records wall-clock runtime_sec on the returned Score's attrs.

Uses stubs (no real model/benchmark) so it runs offline and fast.
"""
import time
import pytest
import brainscore


class _FakeModel:
    identifier = 'fake-model'
    region_layer_map = {}
    in_channels = {'text'}
    out_channels = set()
    required_channels = set()
    output_inputs = {}


class _FakeScore:
    def __init__(self):
        self.attrs = {}


class _FakeBenchmark:
    identifier = 'fake-benchmark'
    required_modalities = set()
    required_input_channels = {'text'}
    requested_output_channels = set()

    def __call__(self, model):
        time.sleep(0.02)          # measurable work
        return _FakeScore()


def test_score_attaches_runtime_sec(monkeypatch):
    monkeypatch.setattr(brainscore, 'load_model', lambda ident: _FakeModel())
    monkeypatch.setattr(brainscore, 'load_benchmark', lambda ident: _FakeBenchmark())
    s = brainscore.score('any-model', 'any-benchmark')
    assert 'runtime_sec' in s.attrs
    assert s.attrs['runtime_sec'] >= 0.02            # at least the sleep
    assert s.attrs['model_identifier'] == 'any-model'
    assert s.attrs['benchmark_identifier'] == 'any-benchmark'


def test_runtime_sec_is_numeric_and_rounded(monkeypatch):
    monkeypatch.setattr(brainscore, 'load_model', lambda ident: _FakeModel())
    monkeypatch.setattr(brainscore, 'load_benchmark', lambda ident: _FakeBenchmark())
    s = brainscore.score('m', 'b')
    rt = s.attrs['runtime_sec']
    assert isinstance(rt, float) and rt == round(rt, 2)


def test_score_accepts_model_and_benchmark_objects(monkeypatch):
    # Passing objects must NOT go through the registry loaders (which would raise
    # here) and must derive the recorded identifiers from the objects themselves.
    def _boom(ident):
        raise AssertionError(f"load should not be called for an object: {ident!r}")
    monkeypatch.setattr(brainscore, 'load_model', _boom)
    monkeypatch.setattr(brainscore, 'load_benchmark', _boom)

    s = brainscore.score(_FakeModel(), _FakeBenchmark())
    assert s.attrs['model_identifier'] == 'fake-model'
    assert s.attrs['benchmark_identifier'] == 'fake-benchmark'
    assert 'runtime_sec' in s.attrs


def test_score_mixes_object_model_with_identifier_benchmark(monkeypatch):
    # One object + one identifier: only the identifier side hits the loader.
    monkeypatch.setattr(brainscore, 'load_model',
                        lambda ident: (_ for _ in ()).throw(
                            AssertionError("model is an object")))
    monkeypatch.setattr(brainscore, 'load_benchmark', lambda ident: _FakeBenchmark())
    s = brainscore.score(_FakeModel(), 'bench-id')
    assert s.attrs['model_identifier'] == 'fake-model'
    assert s.attrs['benchmark_identifier'] == 'bench-id'


def test_score_object_without_identifier_falls_back(monkeypatch):
    class _Anon(_FakeBenchmark):
        identifier = None
    monkeypatch.setattr(brainscore, 'load_model', lambda ident: _FakeModel())
    monkeypatch.setattr(brainscore, 'load_benchmark',
                        lambda ident: (_ for _ in ()).throw(AssertionError("is object")))
    s = brainscore.score('m', _Anon())
    assert s.attrs['benchmark_identifier'] == 'custom-benchmark'  # graceful default


@pytest.mark.parametrize('recorded_name', ['fake-benchmark', 'requested-alias', 'unrelated-benchmark'])
def test_score_identity_must_match_requested_or_resolved_benchmark(monkeypatch, recorded_name):
    class QualifiedBenchmark(_FakeBenchmark):
        def __call__(self, model):
            score = _FakeScore()
            score.attrs['benchmark_identifier'] = recorded_name
            return score

    monkeypatch.setattr(brainscore, 'load_benchmark', lambda ident: QualifiedBenchmark())
    if recorded_name == 'unrelated-benchmark':
        with pytest.raises(ValueError, match='benchmark_identifier'):
            brainscore.score(_FakeModel(), 'requested-alias', check_mem=False)
    else:
        score = brainscore.score(_FakeModel(), 'requested-alias', check_mem=False)
        assert score.attrs['benchmark_identifier'] == recorded_name


class _LabelSubject:
    """Participate-only subject that answers behavior:choice; the last n_outside rows are invalid."""
    in_channels = {'text'}
    out_channels = {'behavior:choice'}
    required_channels = set()
    region_layer_map = {}
    output_inputs = {}
    identifier = 'label-subject'

    def __init__(self, n_outside):
        self.n_outside = n_outside

    def participate(self, session):
        import numpy as np
        from brainscore_core.streaming import StreamEvent
        from brainscore_core.response_formats import response_record
        from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
        while session.next_input() is not None:
            pass
        ids = list(session.stimuli['stimulus_id'])
        rows = np.zeros((len(ids), 2), dtype=np.float32)
        rows[:, 0] = 1.0
        if self.n_outside:
            rows[-self.n_outside:] = np.nan
        session.respond(StreamEvent('behavior:choice', BehavioralAssembly(
            rows, coords={'stimulus_id': ('presentation', ids), 'choice': ['cat', 'dog']},
            dims=['presentation', 'choice'],
            attrs={'response': response_record('behavior:choice', 'test')}), None))

    def reset(self):
        pass


class _LegacyLabelBenchmark:
    identifier = 'legacy-label-benchmark'
    subject_api = 'legacy'              # an old benchmark: it calls look_at on the facade
    required_input_channels = {'text'}
    requested_output_channels = {'behavior:choice'}

    def __call__(self, model):
        import pandas as pd
        from brainscore_core.metrics import Score
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
        stimuli = StimulusSet(pd.DataFrame({'stimulus_id': [f's{i}' for i in range(20)],
                                            'sentence': ['word'] * 20}))
        stimuli.identifier = 'unified-entry-label'
        model.start_task('label', ['cat', 'dog'])
        model.look_at(stimuli)
        return Score(0.5)


def test_the_score_of_an_old_label_benchmark_carries_the_replaced_answer_count(monkeypatch):
    monkeypatch.setattr(brainscore, 'load_model', lambda ident: _LabelSubject(1))
    monkeypatch.setattr(brainscore, 'load_benchmark', lambda ident: _LegacyLabelBenchmark())
    with pytest.warns(UserWarning, match='1 invalid answers'):
        s = brainscore.score('label-subject', 'legacy-label-benchmark', check_mem=False)
    assert s.attrs['n_invalid'] == 1
    assert 'random label' in s.attrs['answer_policy']


def test_the_score_carries_a_zero_when_no_answer_was_replaced(monkeypatch):
    monkeypatch.setattr(brainscore, 'load_model', lambda ident: _LabelSubject(0))
    monkeypatch.setattr(brainscore, 'load_benchmark', lambda ident: _LegacyLabelBenchmark())
    s = brainscore.score('label-subject', 'legacy-label-benchmark', check_mem=False)
    assert s.attrs['n_invalid'] == 0
    assert 'random label' in s.attrs['answer_policy']
