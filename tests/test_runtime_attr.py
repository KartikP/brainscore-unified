"""score() records wall-clock runtime_sec on the returned Score's attrs.

Uses stubs (no real model/benchmark) so it runs offline and fast.
"""
import time
import brainscore


class _FakeScore:
    def __init__(self):
        self.attrs = {}


class _FakeBenchmark:
    def __call__(self, model):
        time.sleep(0.02)          # measurable work
        return _FakeScore()


def test_score_attaches_runtime_sec(monkeypatch):
    monkeypatch.setattr(brainscore, 'load_model', lambda ident: object())
    monkeypatch.setattr(brainscore, 'load_benchmark', lambda ident: _FakeBenchmark())
    s = brainscore.score('any-model', 'any-benchmark')
    assert 'runtime_sec' in s.attrs
    assert s.attrs['runtime_sec'] >= 0.02            # at least the sleep
    assert s.attrs['model_identifier'] == 'any-model'
    assert s.attrs['benchmark_identifier'] == 'any-benchmark'


def test_runtime_sec_is_numeric_and_rounded(monkeypatch):
    monkeypatch.setattr(brainscore, 'load_model', lambda ident: object())
    monkeypatch.setattr(brainscore, 'load_benchmark', lambda ident: _FakeBenchmark())
    s = brainscore.score('m', 'b')
    rt = s.attrs['runtime_sec']
    assert isinstance(rt, float) and rt == round(rt, 2)
