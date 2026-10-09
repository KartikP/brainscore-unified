"""Registration delegates evaluation and preserves failures and raw scores."""
import json
import math
from collections import Counter

import pytest

import brainscore
from brainscore.benchmarks.coggym import EXPERIMENTS
from brainscore.benchmarks.coggym.benchmark import CogGymBenchmark
from brainscore.experiments import Experiment, RecordInputsOutputs
from brainscore.harnesses import coggym
from brainscore_core.compatibility import CompatibilityError

from .test_coggym import reference, make_subject, provider

pytestmark = pytest.mark.unit


@pytest.fixture
def scored_reference(reference, monkeypatch):
    root, cli = reference
    _, analysis = coggym._load_reference(root)
    def analyze(paths, *args, **kwargs):
        runs = [json.loads(path.read_text()) for path in paths]
        row = {'model': runs[0]['model'], 'experiment': runs[0]['experiment'],
               'r2': 0.375, 'n_items': 3, 'n_artifacts': len(runs)}
        return [row], [], []
    analysis.analyze = analyze
    return root, analysis


def test_all_public_experiments_registered_lazily():
    ids = {name for name in brainscore.benchmark_registry if name.startswith('CogGym.')}
    assert ids == {f'CogGym.{path.replace("/", ".")}' for path in EXPERIMENTS}
    assert Counter(EXPERIMENTS.values()) == {'text': 11, 'image': 6, 'video': 6}


def test_loading_requires_explicit_checkout_before_model_use(monkeypatch):
    monkeypatch.delenv('BRAINSCORE_COGGYM_CHECKOUT', raising=False)
    with pytest.raises(ValueError, match='BRAINSCORE_COGGYM_CHECKOUT'):
        brainscore.load_benchmark('CogGym.Hu2023Fine.exp1')


def test_environment_checkout_and_explicit_provider_reset(scored_reference, monkeypatch):
    root, _ = scored_reference
    monkeypatch.setenv('BRAINSCORE_COGGYM_CHECKOUT', str(root))
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1')
    with pytest.raises(TypeError, match='provider.reset'):
        benchmark.protocol(model='fixture')
    assert math.isnan(float(benchmark.ceiling))


def test_registered_scoring_preserves_raw_score_and_repetitions(scored_reference, tmp_path, monkeypatch):
    root, _ = scored_reference
    resets, calls = [], []
    def capture(request):
        calls.append(request)
        return provider(request)
    benchmark = brainscore.load_benchmark(
        'CogGym.Hu2023Fine.exp1', checkout=root, reset=resets.append,
        repetitions=2, max_tokens=1234, output_dir=tmp_path / 'scored',
    )
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path / 'cache'))
    score = brainscore.score(make_subject(capture), benchmark, check_mem=False)
    assert float(score) == float(score.attrs['raw']) == 0.375
    assert score.attrs['normalized'] is False
    assert score.attrs['ceiling_status'] == 'not_estimated'
    assert score.attrs['configuration']['repetitions'] == 2
    assert resets == [1, 2]
    assert len(calls) == 6
    assert all(call['max_tokens'] == 1234 for call in calls)
    assert score.attrs['benchmark_identifier'] == 'CogGym.Hu2023Fine.exp1'
    assert (tmp_path / 'scored/coggym/analysis.json').is_file()


def test_protocol_and_plain_runner_produce_identical_reference_artifacts(scored_reference, tmp_path):
    root, _ = scored_reference
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', checkout=root)
    direct = coggym.CogGymRunner(root, experiment='Hu2023Fine/exp1', model='fixture', max_tokens=8192)
    from types import SimpleNamespace
    def complete(system, messages, model, temperature, max_tokens):
        return provider(dict(system=system, messages=messages, model=model,
                             temperature=temperature, max_tokens=max_tokens))
    direct.run(SimpleNamespace(complete_with_metadata=complete),
               output_dir=tmp_path / 'direct', reset=lambda repetition: None)
    result = Experiment(
        subject=make_subject(provider),
        protocol=benchmark.protocol(model='fixture', reset=lambda repetition: None),
        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'registered',
    ).run()
    plain = json.loads((tmp_path / 'direct/run-001.json').read_text())
    wrapped = json.loads((result.directory / 'coggym/run-001.json').read_text())
    assert wrapped == plain
    assert float(result.value) == 0.375
    assert sum(event['kind'] == 'input' for event in result.record.events()) == 3


@pytest.mark.parametrize('value', [None, float('nan'), float('inf')])
def test_undefined_score_fails_and_preserves_results(scored_reference, tmp_path, value):
    root, analysis = scored_reference
    # None is serialized upstream; nonfinite values also must never become scores.
    analysis.analyze = lambda *args, **kwargs: ([{
        'model': 'fixture', 'experiment': 'Hu2023Fine/exp1', 'r2': value,
    }], [], [])
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', checkout=root,
        reset=lambda repetition: None, output_dir=tmp_path / 'run')
    with pytest.raises(ValueError):
        benchmark(make_subject(provider))
    assert (tmp_path / 'run/coggym/run-001.json').is_file()
    assert json.loads((tmp_path / 'run/experiment.json').read_text())['status'] == 'failed'


def test_feature_only_model_rejected_before_requests(scored_reference):
    root, _ = scored_reference
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', checkout=root,
        reset=lambda repetition: None)
    with pytest.raises(CompatibilityError):
        benchmark(brainscore.load_model('chance-baseline'))


def test_configuration_is_not_discarded_on_domain_fallback():
    with pytest.raises(TypeError, match='unified registry'):
        brainscore.load_benchmark('some-domain-benchmark', repetitions=2)


def test_video_registration_preserves_still_image_instructions(reference):
    root, cli = reference
    (root / 'evaluation/public_manifest.json').write_text(json.dumps({
        'experiments': [{'path': 'Fu2025Hierarchical/exp1'}],
    }))
    cli.build_messages = lambda trial, experiment: ('system', [{
        'role': 'user', 'content': [{'type': 'text', 'text': 'instructions'},
                                 {'type': 'image'}, {'type': 'video'}],
    }])
    benchmark = brainscore.load_benchmark('CogGym.Fu2025Hierarchical.exp1', checkout=root)
    protocol = benchmark.protocol(model='fixture', reset=lambda repetition: None)
    assert set(protocol.metadata['modalities']) == {'text', 'image', 'video'}
