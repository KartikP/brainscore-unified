"""All six registered factories exercised with tiny CPU models/synthetic data.

Metrics and data loading are replaced. These are route/activation tests, not
measurements of neural predictivity on MajajHong or Pereira.
"""
import functools

import numpy as np
import pytest
import torch
from PIL import Image

from brainscore_core.metrics import Score
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore.validation.benchmark_parity import (
    CASES, PEREIRA_CASES, PEREIRA_NATIVE_ULPS, PEREIRA_SCORE_ATOL,
    assert_pereira_activations, assert_pereira_score_drift, fp32_ulp_tolerance, validate_case,
)
from tests.test_umi_review_regressions import language_pair, _stimuli

pytestmark = pytest.mark.integration


def _metric(source, target):
    return Score(float(np.mean(np.abs(source.values))))


@pytest.fixture
def synthetic_benchmarks(monkeypatch, tmp_path):
    from brainscore_vision.benchmarks.majajhong2015 import benchmark as vision_legacy
    from brainscore_vision.benchmarks.majajhong2015 import unified as vision_unified
    from brainscore_language.benchmarks.pereira2018.benchmark import _Pereira2018Experiment
    import brainscore_language.benchmarks.pereira2018.benchmark as language_benchmark
    stimuli = _stimuli()
    stimuli['object_name'] = ['cat', 'cat', 'dog', 'dog']
    paths = {}
    for i, sid in enumerate(stimuli.stimulus_id):
        path = tmp_path / f'{sid}.png'
        Image.new('RGB', (8, 8), (20 + i * 30, 40, 60)).save(path)
        paths[sid] = path
    stimuli.stimulus_paths = paths
    stimuli['image_path'] = [str(paths[sid]) for sid in stimuli.stimulus_id]
    seen = []
    def vision_data(average_repetitions, region, access, **kwargs):
        seen.append((region, access))
        data = NeuroidAssembly(np.zeros((4, 1)), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', list(stimuli.stimulus_id)),
            'object_name': ('presentation', list(stimuli.object_name)),
            'neuroid_id': ('neuroid', ['target']), 'region': ('neuroid', [region]),
            'time_bin_start': 70, 'time_bin_end': 170,
        })
        data = data.expand_dims('time_bin').squeeze('time_bin')
        # A scalar tuple matches the single-time-bin legacy data contract.
        scalar = np.empty((), dtype=object)
        scalar[()] = (70, 170)
        data['time_bin'] = scalar
        data.attrs['stimulus_set'] = stimuli.drop(columns='sentence')
        data.attrs['stimulus_set'].identifier = stimuli.identifier
        data.attrs['stimulus_set'].stimulus_paths = paths
        return data
    for module in (vision_legacy, vision_unified):
        monkeypatch.setattr(module, 'load_assembly', vision_data)
        monkeypatch.setattr(module, 'load_metric', lambda *a, **k: _metric)
        monkeypatch.setattr(module, 'load_ceiling', lambda *a, **k: lambda data: Score(1))
    def language_data(self, experiment, atlas):
        seen.append(experiment)
        return NeuroidAssembly(np.zeros((4, 1)), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', list(stimuli.stimulus_id)),
            'stimulus': ('presentation', list(stimuli.sentence)),
            'passage_label': ('presentation', ['b', 'b', 'a', 'a']),
            'passage_index': ('presentation', [0, 1, 0, 1]),
            'story': ('presentation', ['b', 'b', 'a', 'a']),
            'neuroid_id': ('neuroid', ['target']),
        })
    monkeypatch.setattr(_Pereira2018Experiment, '_load_data', language_data)
    monkeypatch.setattr(_Pereira2018Experiment, '_load_ceiling', lambda *a, **k: Score(1))
    monkeypatch.setattr(language_benchmark, 'load_metric', lambda *a, **k: _metric)
    return seen


def _vision_candidate(case, route):
    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
    from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
    from brainscore_vision.compat.unified_adapter import VisionModelAdapter
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 4)).eval()
    mapping = {'V4': '1', 'IT': '1'}
    wrapper = PytorchWrapper(model, functools.partial(load_preprocess_images, image_size=8),
                             identifier='synthetic-parity', batch_size=4)
    if route == 'native':
        return BrainScoreModel('synthetic-native', model=model, region_layer_map=mapping,
                               preprocessors={'vision': wrapper}, visual_degrees=8)
    legacy = ModelCommitment('synthetic-legacy', wrapper, ['1'], region_layer_map=mapping, visual_degrees=8)
    return VisionModelAdapter(legacy) if route == 'adapter' else legacy


@pytest.mark.parametrize('case', CASES, ids=lambda case: case.unified)
def test_all_registered_factories_on_both_routes(case, synthetic_benchmarks, language_pair):
    native, adapter = language_pair
    def factory(case, route):
        if case.domain == 'vision':
            return _vision_candidate(case, route)
        native.reset()
        adapter.reset()
        return {'native': native, 'adapter': adapter, 'legacy': adapter._legacy}[route]
    report = validate_case(case, factory)
    assert set(report['routes']) == {'legacy', 'adapter', 'native'}
    assert report['routes']['adapter']['max_activation_delta'] == 0
    if case.domain == 'vision':
        region = 'V4' if '.V4-' in case.legacy else 'IT'
        access = 'public' if 'public' in case.legacy else 'private'
        assert set(synthetic_benchmarks) == {(region, access)}
    else:
        experiment = case.legacy.split('.')[1].split('-')[0]
        assert set(synthetic_benchmarks) == {experiment}


def test_harness_rejects_adapter_disguised_as_native(synthetic_benchmarks, language_pair):
    _, adapter = language_pair
    def factory(case, route):
        adapter.reset()
        return adapter._legacy if route == 'legacy' else adapter
    with pytest.raises(TypeError, match='native route'):
        validate_case(CASES[-1], factory)


def test_harness_detects_missing_native_context(synthetic_benchmarks, language_pair, monkeypatch):
    native, adapter = language_pair
    wrapper = native._preprocessors['text']
    monkeypatch.setattr(wrapper, '_extract_texts', lambda stimuli: list(stimuli.sentence))
    def factory(case, route):
        native.reset()
        adapter.reset()
        return {'native': native, 'adapter': adapter, 'legacy': adapter._legacy}[route]
    with pytest.raises(AssertionError, match='native activations differ'):
        validate_case(CASES[-1], factory)


@pytest.fixture
def run_parity_observations(language_pair):
    """Exercise numeric acceptance using controlled metric inputs and scores.

    Extraction is covered above. This fixture uses no real benchmark metric,
    weights or data, and permits score changes independent of activation changes.
    """
    native, adapter = language_pair

    def run(case, changes, **kwargs):
        active_route = None

        def factory(case, route):
            nonlocal active_route
            active_route = route
            native.reset()
            adapter.reset()
            return {'native': native, 'adapter': adapter, 'legacy': adapter._legacy}[route]

        class Benchmark:
            metric = staticmethod(lambda source, target: Score(0.25))
            _similarity_metric = metric
            ceiling = Score(changes.get('legacy', {}).get('ceiling', 0.5))

            def __call__(self, candidate):
                observation = changes.get(active_route, {})
                source = NeuroidAssembly(
                    np.array([[1., 2.]]) + observation.get('activation_delta', 0),
                    dims=['presentation', 'neuroid'], coords={
                        'stimulus_id': ('presentation', ['s0']),
                        'stimulus': ('presentation', ['synthetic']),
                        'neuroid_id': ('neuroid', ['n0', 'n1']),
                        'unit_index': ('neuroid', [0, 1]),
                    })
                metric = self._similarity_metric if case.domain == 'vision' else self.metric
                metric(source, source.copy(deep=True))
                score = Score(observation.get('score', 0.5))
                score.attrs['raw'] = Score(observation.get('raw', 0.25))
                return score

        return validate_case(case, factory, benchmark_loader=lambda identifier: Benchmark(), **kwargs)

    return run


@pytest.mark.parametrize('case', CASES, ids=lambda case: case.unified)
@pytest.mark.parametrize('field', ['activation_delta', 'score', 'raw'])
def test_adapter_exactness_cannot_be_relaxed_by_native_tolerances(case, field, run_parity_observations):
    baseline = {'activation_delta': 0, 'score': 0.5, 'raw': 0.25}[field]
    message = 'activations' if field == 'activation_delta' else field
    with pytest.raises(AssertionError, match=f'adapter {message} differ'):
        run_parity_observations(case, {'adapter': {field: baseline + 1e-12}},
                                native_atol=1, score_atol=1)


@pytest.mark.parametrize('case, legacy_score, native_score, legacy_raw, native_raw', [
    (CASES[-2], 0.8711741877922449, 0.8723817692825759,
     0.30821208826023067, 0.3086393176456909),
    (CASES[-1], 0.8280753625165158, 0.8277164923953866,
     0.30095362369641643, 0.30082319684364545),
])
def test_native_accepts_measured_score_gaps_and_reports_ulp_policy(
        case, legacy_score, native_score, legacy_raw, native_raw, run_parity_observations):
    reference = dict(score=legacy_score, raw=legacy_raw, ceiling=legacy_raw / legacy_score)
    # Synthetic activation discrepancy exercises the small-coordinate floor.
    # These are not the original GPU activation arrays.
    changed = dict(score=native_score, raw=native_raw, activation_delta=1e-4)
    report = run_parity_observations(case, {'legacy': reference, 'adapter': reference, 'native': changed})
    assert report['native_atol'] is None
    assert report['native_ulps'] == 16
    assert report['native_ulp_magnitude_floor'] == 64
    assert report['score_atol'] == 0.002
    assert report['routes']['native']['score'] == native_score


@pytest.mark.parametrize('field', ['score', 'raw'])
def test_native_score_guard_rejects_drift_with_identical_activations(field, run_parity_observations):
    baseline = {'score': 0.5, 'raw': 0.25}[field]
    with pytest.raises(AssertionError, match=f'native {field} differs'):
        run_parity_observations(CASES[-1], {'native': {field: baseline + 0.003}})


def test_native_activation_guard_remains_when_scores_match(run_parity_observations):
    with pytest.raises(AssertionError, match='native activations differ'):
        run_parity_observations(CASES[-1], {'native': {'activation_delta': 2 ** -12}})


def test_native_reports_score_and_activation_failures_together(run_parity_observations):
    with pytest.raises(AssertionError) as failure:
        run_parity_observations(CASES[-1], {'native': {
            'score': 0.503, 'raw': 0.253, 'activation_delta': 2 ** -12}})
    message = str(failure.value)
    assert message.index('native score differs') < message.index('native raw differs')
    assert message.index('native raw differs') < message.index('native activations differ')


@pytest.mark.parametrize('reference, delta, local_ulps', [
    (273.237, 2 ** -13, 4), (278.512, 2 ** -13, 4),
    (1.619, 7.927417755126953e-5, 665), (99.0666, 8.392333984375e-5, 11),
    (125.896, 9.1552734375e-5, 12), (-134.127, 6.103515625e-5, 4),
])
def test_reported_binades_and_deltas_fit_policy(reference, delta, local_ulps):
    # Reconstruct controlled perturbations from the supplied magnitudes/deltas;
    # the original coordinate values are rounded and GPU arrays are unavailable.
    reference = np.array([reference], dtype=np.float32)
    assert delta / float(np.spacing(np.abs(reference[0]))) == local_ulps
    actual = reference.astype(np.float64) + delta
    assert assert_pereira_activations(actual, reference) <= PEREIRA_NATIVE_ULPS
    assert 2 ** -13 == 4 * 2 ** -15


@pytest.mark.parametrize('reference', [0., 1.619, -1.619, 64., 273.237, -278.512, 1024.])
def test_ulp_limit_accepts_boundary_and_rejects_one_more_effective_ulp(reference):
    reference = np.array([reference], dtype=np.float32)
    limit = fp32_ulp_tolerance(reference)
    assert_pereira_activations(reference.astype(np.float64) + limit, reference)
    with pytest.raises(AssertionError, match='effective FP32 ULPs exceeds 16'):
        assert_pereira_activations(reference.astype(np.float64) + limit * 17 / 16, reference)


def test_ulp_limit_is_per_element_not_global_maximum():
    reference = np.array([0., 273.237], dtype=np.float32)
    assert_pereira_activations(reference.astype(np.float64) + [0., 2 ** -12], reference)
    with pytest.raises(AssertionError):
        assert_pereira_activations(reference.astype(np.float64) + [2 ** -12, 0.], reference)


@pytest.mark.parametrize('actual, reference', [([np.nan], [0.]), ([np.inf], [0.]),
                                             ([0.], [np.nan]), ([0.], [np.inf]),
                                             ([0., 0.], [0.])])
def test_ulp_check_rejects_nonfinite_or_mismatched_shapes(actual, reference):
    with pytest.raises(AssertionError):
        assert_pereira_activations(actual, reference)


def test_pereira_raw_score_uses_ceiling_units_even_when_published_score_is_clipped(run_parity_observations):
    # Equal clipped scores must not hide a >0.002 normalized raw-score drift.
    reference = dict(score=1., raw=0.3, ceiling=0.25)
    with pytest.raises(AssertionError, match='native raw differs'):
        run_parity_observations(CASES[-1], {
            'legacy': reference, 'adapter': reference,
            'native': dict(score=1., raw=0.3006)})


def test_vision_native_keeps_original_absolute_tolerance(run_parity_observations):
    with pytest.raises(AssertionError, match='native activations differ'):
        run_parity_observations(CASES[0], {'native': {'activation_delta': 2e-6}})


def _paired_score_reports(deltas, raw_deltas=None):
    raw_deltas = deltas if raw_deltas is None else raw_deltas
    return [dict(benchmark=case.unified, routes={
        'legacy': dict(score=0.5, raw=0.25, ceiling=0.5),
        'native': dict(score=0.5 + delta, raw=0.25 + raw_delta * 0.5, ceiling=0.5),
    }) for case, delta, raw_delta in zip(PEREIRA_CASES, deltas, raw_deltas)]


@pytest.mark.parametrize('deltas', [(0., 0.), (0.0012075814903310, -0.0003588701211292),
                                  (0.0012, -0.0004)])
def test_paired_score_guard_accepts_measured_or_exact_parity(deltas):
    assert_pereira_score_drift(_paired_score_reports(deltas))


@pytest.mark.parametrize('deltas', [(0.0001, 0.0001), (-0.0001, -0.0001)])
@pytest.mark.parametrize('key', ['score', 'raw'])
def test_paired_score_guard_rejects_same_sign_bias_inside_individual_tolerances(deltas, key):
    assert all(abs(delta) < PEREIRA_SCORE_ATOL for delta in deltas)
    reports = _paired_score_reports(deltas if key == 'score' else (0., 0.),
                                    deltas if key == 'raw' else (0., 0.))
    with pytest.raises(AssertionError, match=f'{key}: same-sign drift'):
        assert_pereira_score_drift(reports)


@pytest.mark.parametrize('key', ['score', 'raw'])
def test_paired_score_guard_detects_growing_drift_before_global_budget(key):
    # A monotone outward drift still inside 0.002 crosses the frozen rounding
    # envelope. Growth smaller than the envelope's resolution is not claimed
    # to be detectable by any finite tolerance.
    for deltas in ((0.0012, -0.0004), (0.00124, -0.00044)):
        assert_pereira_score_drift(_paired_score_reports(deltas))
    deltas = (0.0013, -0.0005)
    assert all(abs(delta) < PEREIRA_SCORE_ATOL for delta in deltas)
    reports = _paired_score_reports(deltas if key == 'score' else (0., 0.),
                                    deltas if key == 'raw' else (0., 0.))
    with pytest.raises(AssertionError, match=f'{key} drift .*historical rounding envelope'):
        assert_pereira_score_drift(reports)


@pytest.mark.parametrize('indices', [[0], [1], [0, 0]])
def test_paired_score_guard_requires_both_cases(indices):
    reports = _paired_score_reports((0., 0.))
    with pytest.raises(AssertionError, match='both benchmarks exactly once'):
        assert_pereira_score_drift([reports[index] for index in indices])


def test_slow_entrypoint_actually_enforces_joint_bias_guard(monkeypatch, tmp_path):
    from tests import test_benchmark_parity_slow as slow
    reports = {report['benchmark']: report for report in _paired_score_reports((0.0001, 0.0001))}
    monkeypatch.setattr(slow, '_require_checkpoint', lambda domain: None)
    monkeypatch.setattr(slow, 'validate_case', lambda case, factory: reports[case.unified])
    with pytest.raises(AssertionError, match='same-sign drift'):
        slow.test_pereira_243_and_384_score_parity(tmp_path)
