"""score() records wall-clock runtime_sec on the returned Score's attrs.

Uses stubs (no real model/benchmark) so it runs offline and fast.
"""
import time
import brainscore


class _FakeModel:
    identifier = 'fake-model'
    supported_modalities = set()
    available_modalities = set()
    required_modalities = set()
    region_layer_map = {}
    in_channels = set()
    out_channels = set()
    required_channels = set()


class _FakeScore:
    def __init__(self):
        self.attrs = {}


class _FakeBenchmark:
    identifier = 'fake-benchmark'
    required_modalities = set()
    required_input_channels = set()
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
