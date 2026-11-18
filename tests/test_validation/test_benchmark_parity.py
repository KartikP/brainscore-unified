"""Every release case exercised with tiny CPU models/synthetic data.

Metrics and data loading are replaced. These are route/activation tests, not
measurements of neural predictivity on MajajHong or Pereira.
"""
import functools

import numpy as np
import pytest
import torch
from PIL import Image

from brainscore_core.metrics import Score
from brainscore import BrainScoreModel
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore.validation.benchmark_parity import (
    ADAPTER_SCORE_ULPS, CASES, L4_NATIVE_ULPS, RELEASE_CASES, RIDGE_CASES,
    adapter_score_tolerance,
    assert_pereira_activations, assert_pereira_score_drift, fp32_ulp_tolerance, validate_case,
)
from tests.test_package.test_umi_review_regressions import language_pair, _stimuli

pytestmark = pytest.mark.integration

# The tolerance guards are domain-independent code. One vision and one language
# case cover both routes; the factory test below still runs every release case.
GUARD_CASES = (CASES[0], RIDGE_CASES[0])


@pytest.mark.parametrize('failure', [False, True])
def test_snapshot_cleanup_and_hash_compatibility(tmp_path, run_parity_observations, failure):
    changes = {'native': {'activation_delta': 1.}} if failure else {}
    if failure:
        with pytest.raises(AssertionError, match='activations differ'):
            run_parity_observations(CASES[0], changes, scratch_dir=tmp_path)
    else:
        import hashlib
        report = run_parity_observations(CASES[0], changes, scratch_dir=tmp_path)
        expected_hash = hashlib.sha256(np.array([[1., 2.]])).hexdigest()
        for route in ('legacy', 'adapter', 'native'):
            assert report['routes'][route]['metric_inputs'][0]['values_sha256'] == expected_hash
    assert not list(tmp_path.iterdir())


def _metric(source, target):
    return Score(float(np.mean(np.abs(source.values))))


@pytest.fixture
def constant_metric(monkeypatch):
    # Failure-isolation tests still capture and compare real extracted values,
    # but their scalar metric cannot fail first due to layout-dependent sums.
    monkeypatch.setattr(__name__ + '._metric', lambda source, target: Score(0.25))


@pytest.fixture
def synthetic_benchmarks(monkeypatch, tmp_path):
    from brainscore_vision.benchmarks.majajhong2015 import benchmark as vision_legacy
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
    monkeypatch.setattr(vision_legacy, 'load_assembly', vision_data)
    monkeypatch.setattr(vision_legacy, 'load_metric', lambda *a, **k: _metric)
    monkeypatch.setattr(vision_legacy, 'load_ceiling', lambda *a, **k: lambda data: Score(1))
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


def _vision_subject(case, route):
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
                               extractor={'vision': wrapper}, visual_degrees=8)
    legacy = ModelCommitment('synthetic-legacy', wrapper, ['1'], region_layer_map=mapping, visual_degrees=8)
    return VisionModelAdapter(legacy) if route == 'adapter' else legacy


@pytest.mark.parametrize('case', RELEASE_CASES, ids=lambda case: case.legacy)
def test_all_registered_factories_on_both_routes(case, synthetic_benchmarks, language_pair):
    native, adapter = language_pair
    def factory(case, route):
        if case.domain == 'vision':
            return _vision_subject(case, route)
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


def test_harness_rejects_adapter_disguised_as_native(synthetic_benchmarks, language_pair, constant_metric):
    _, adapter = language_pair
    def factory(case, route):
        adapter.reset()
        return adapter._legacy if route == 'legacy' else adapter
    with pytest.raises(TypeError, match='native route'):
        validate_case(RIDGE_CASES[-1], factory)


def test_harness_detects_missing_native_context(synthetic_benchmarks, language_pair, constant_metric, monkeypatch):
    native, adapter = language_pair
    wrapper = native.extractor['text']
    def factory(case, route):
        native.reset()
        adapter.reset()
        return {'native': native, 'adapter': adapter, 'legacy': adapter._legacy}[route]
    # First establish that the otherwise-identical setup passes with context.
    validate_case(RIDGE_CASES[-1], factory)
    monkeypatch.setattr(wrapper, '_table_inputs', lambda stimuli: list(stimuli.sentence))
    with pytest.raises(AssertionError, match='native activations differ') as failure:
        validate_case(RIDGE_CASES[-1], factory)
    assert 'score differs' not in str(failure.value)
    assert 'raw differs' not in str(failure.value)


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

            def __call__(self, subject):
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


@pytest.mark.parametrize('case', GUARD_CASES, ids=lambda case: case.legacy)
@pytest.mark.parametrize('field', ['activation_delta', 'score', 'raw'])
def test_adapter_guards_cannot_be_relaxed_by_native_tolerances(case, field, run_parity_observations):
    baseline = {'activation_delta': 0, 'score': 0.5, 'raw': 0.25}[field]
    message = 'activations' if field == 'activation_delta' else field
    delta = 1e-12 if field == 'activation_delta' else 5 * float(np.spacing(np.float32(baseline)))
    with pytest.raises(AssertionError, match=f'adapter {message} differ'):
        run_parity_observations(case, {'adapter': {field: baseline + delta}},
                                native_atol=1, score_atol=1)


@pytest.mark.parametrize('case', GUARD_CASES, ids=lambda case: case.legacy)
@pytest.mark.parametrize('field', ['score', 'raw'])
def test_adapter_scores_allow_four_fp32_ulps_with_exact_activations(case, field, run_parity_observations):
    # Controlled scalar perturbations, not invented CI activation arrays.
    # Cover the CI binade and both directions, including the four-ULP boundary.
    reference = float(np.float32(0.03181))
    spacing = float(np.spacing(np.float32(reference)))
    assert spacing == 2 ** -28
    assert adapter_score_tolerance(reference) == ADAPTER_SCORE_ULPS * spacing
    for ulps in (1, -4, 4):
        observed = run_parity_observations(case, {
            'legacy': {field: reference},
            'adapter': {field: reference + ulps * spacing},
            'native': {field: reference},
        })
        assert observed['routes']['adapter']['max_activation_delta'] == 0
    # A genuine activation change must fail even with an allowed scalar delta.
    with pytest.raises(AssertionError, match='adapter activations differ'):
        run_parity_observations(case, {
            'legacy': {field: reference},
            'adapter': {field: reference + spacing, 'activation_delta': 1e-12},
            'native': {field: reference},
        })


@pytest.mark.parametrize('reference', [0., 1e-50, 1e-40, -0.03181, 0.5, 1., np.finfo(np.float32).max])
def test_adapter_score_tolerance_scales_without_an_absolute_floor(reference):
    magnitude = np.float32(abs(reference))
    # Compute spacing independently, using the preceding binade at max_float32
    # to avoid nextafter overflowing to infinity.
    if magnitude == np.finfo(np.float32).max:
        spacing = float(magnitude) - float(np.nextafter(magnitude, np.float32(0)))
    else:
        spacing = float(np.nextafter(magnitude, np.float32(np.inf))) - float(magnitude)
    assert adapter_score_tolerance(reference) == 4 * spacing


@pytest.mark.parametrize('field', ['score', 'raw'])
def test_native_score_guard_rejects_drift_with_identical_activations(field, run_parity_observations):
    baseline = {'score': 0.5, 'raw': 0.25}[field]
    with pytest.raises(AssertionError, match=f'native {field} differs'):
        run_parity_observations(RIDGE_CASES[-1], {'native': {field: baseline + 0.003}})


def test_native_activation_guard_remains_when_scores_match(run_parity_observations):
    with pytest.raises(AssertionError, match='native activations differ'):
        run_parity_observations(RIDGE_CASES[-1], {'native': {'activation_delta': 2 ** -12}})


def test_native_reports_score_and_activation_failures_together(run_parity_observations):
    with pytest.raises(AssertionError) as failure:
        run_parity_observations(RIDGE_CASES[-1], {'native': {
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
    assert assert_pereira_activations(actual, reference, ulps=L4_NATIVE_ULPS) <= L4_NATIVE_ULPS
    assert 2 ** -13 == 4 * 2 ** -15


@pytest.mark.parametrize('reference', [0., 1.619, -1.619, 64., 273.237, -278.512, 1024.])
def test_ulp_limit_accepts_boundary_and_rejects_one_more_effective_ulp(reference):
    reference = np.array([reference], dtype=np.float32)
    limit = fp32_ulp_tolerance(reference, ulps=L4_NATIVE_ULPS)
    assert_pereira_activations(reference.astype(np.float64) + limit, reference, ulps=L4_NATIVE_ULPS)
    with pytest.raises(AssertionError, match='effective FP32 ULPs exceeds 16'):
        assert_pereira_activations(reference.astype(np.float64) + limit * 17 / 16, reference,
                                   ulps=L4_NATIVE_ULPS)


def test_ulp_limit_is_per_element_not_global_maximum():
    reference = np.array([0., 273.237], dtype=np.float32)
    assert_pereira_activations(reference.astype(np.float64) + [0., 2 ** -12], reference, ulps=L4_NATIVE_ULPS)
    with pytest.raises(AssertionError):
        assert_pereira_activations(reference.astype(np.float64) + [2 ** -12, 0.], reference, ulps=L4_NATIVE_ULPS)


@pytest.mark.parametrize('actual, reference', [([np.nan], [0.]), ([np.inf], [0.]),
                                             ([0.], [np.nan]), ([0.], [np.inf]),
                                             ([0., 0.], [0.])])
def test_ulp_check_rejects_nonfinite_or_mismatched_shapes(actual, reference):
    with pytest.raises(AssertionError):
        assert_pereira_activations(actual, reference, ulps=L4_NATIVE_ULPS)


def test_vision_native_keeps_original_absolute_tolerance(run_parity_observations):
    with pytest.raises(AssertionError, match='native activations differ'):
        run_parity_observations(CASES[0], {'native': {'activation_delta': 2e-6}})


def _paired_score_reports(deltas, raw_deltas=None):
    raw_deltas = deltas if raw_deltas is None else raw_deltas
    return [dict(benchmark=case.legacy, routes={
        'legacy': dict(score=0.5, raw=0.25, ceiling=0.5),
        'native': dict(score=0.5 + delta, raw=0.25 + raw_delta * 0.5, ceiling=0.5),
    }) for case, delta, raw_delta in zip(RIDGE_CASES, deltas, raw_deltas)]


def test_paired_score_guard_needs_an_fp32_policy():
    from brainscore.validation.benchmark_parity import HISTORICAL_POLICY
    with pytest.raises(ValueError, match='FP32 policy'):
        assert_pereira_score_drift(_paired_score_reports((0., 0.)), policy=HISTORICAL_POLICY)


@pytest.mark.parametrize('case', RIDGE_CASES)
def test_cpu_budget_is_explicit_and_preserves_failure_evidence(case, run_parity_observations):
    from brainscore.validation.benchmark_parity import CPU_FP32_POLICY, ParityFailure
    changes = {'native': {'activation_delta': 32 * 2 ** -17, 'score': .500002}}
    with pytest.raises(ParityFailure) as failure:
        run_parity_observations(case, changes)
    assert failure.value.report['routes']['native']['score'] == .500002
    assert failure.value.report['routes']['native']['max_activation_delta'] == 32 * 2 ** -17
    assert failure.value.report['routes']['native']['metric_inputs'][0]['shape'] == [1, 2]
    report = run_parity_observations(case, changes, policy=CPU_FP32_POLICY)
    assert report['native_ulps'] == 32
    assert report['score_atol'] == 1e-5
    with pytest.raises(ParityFailure, match='exceeds 32'):
        run_parity_observations(case, {'native': {'activation_delta': 33 * 2 ** -17}}, policy=CPU_FP32_POLICY)


@pytest.mark.parametrize('field', ['activation_delta', 'score', 'raw'])
def test_cpu_profile_does_not_relax_adapter(field, run_parity_observations):
    from brainscore.validation.benchmark_parity import CPU_FP32_POLICY, RIDGE_CASES
    value = {'activation_delta': 1e-12, 'score': .500001, 'raw': .250001}[field]
    for case in RIDGE_CASES:
        with pytest.raises(AssertionError, match='adapter'):
            run_parity_observations(case, {'adapter': {field: value}}, policy=CPU_FP32_POLICY)


def test_cpu_ridge_budget_checks_unclipped_raw_score(run_parity_observations):
    from brainscore.validation.benchmark_parity import CPU_FP32_POLICY, RIDGE_CASES
    reference = dict(score=1., raw=.3, ceiling=.25)
    for case in RIDGE_CASES:
        with pytest.raises(AssertionError, match='native raw differs'):
            run_parity_observations(case, {'legacy': reference, 'adapter': reference,
                'native': dict(score=1., raw=.300003)}, policy=CPU_FP32_POLICY)


@pytest.mark.parametrize('key', ['score', 'raw'])
def test_cpu_mean_signed_drift_has_a_finite_budget(key):
    from brainscore.validation.benchmark_parity import CPU_FP32_POLICY
    def reports(deltas):
        return _paired_score_reports(deltas if key == 'score' else (0, 0),
                                     deltas if key == 'raw' else (0, 0))
    assert_pereira_score_drift(reports((2e-6, 1e-6)), policy=CPU_FP32_POLICY)
    with pytest.raises(AssertionError, match='mean signed drift'):
        assert_pereira_score_drift(reports((6e-6, 6e-6)), policy=CPU_FP32_POLICY)
    with pytest.raises(AssertionError, match='per-case budget'):
        assert_pereira_score_drift(reports((11e-6, -11e-6)), policy=CPU_FP32_POLICY)
    with pytest.raises(AssertionError, match='both experiments'):
        assert_pereira_score_drift(reports((0, 0))[:1], policy=CPU_FP32_POLICY)


@pytest.mark.parametrize('case', RIDGE_CASES)
def test_l4_keeps_the_16_ulp_activation_budget_with_tighter_scores(case, run_parity_observations):
    from brainscore.validation.benchmark_parity import L4_FP32_POLICY
    report = run_parity_observations(case, {'native': {'activation_delta': 16 * 2 ** -17,
        'score': .500002}}, policy=L4_FP32_POLICY)
    assert report['native_ulps'] == 16
    assert report['score_atol'] == 1e-5
    with pytest.raises(AssertionError, match='exceeds 16'):
        run_parity_observations(case, {'native': {'activation_delta': 17 * 2 ** -17}}, policy=L4_FP32_POLICY)


def test_l4_applies_same_activation_rule_to_ridge_and_retains_adapter_guard(run_parity_observations):
    from brainscore.validation.benchmark_parity import L4_FP32_POLICY, RIDGE_CASES
    for case in RIDGE_CASES:
        report = run_parity_observations(case, {'native': {'activation_delta': 16 * 2 ** -17}}, policy=L4_FP32_POLICY)
        assert report['native_ulps'] == 16
        assert report['routes']['legacy']['ceiling'] == .5
        with pytest.raises(AssertionError, match='adapter activations differ'):
            run_parity_observations(case, {'adapter': {'activation_delta': 1e-12}}, policy=L4_FP32_POLICY)
        with pytest.raises(AssertionError, match='native raw differs'):
            run_parity_observations(case, {'native': {'raw': .250006}}, policy=L4_FP32_POLICY)


def test_l4_preserves_paired_bias_guard():
    from brainscore.validation.benchmark_parity import L4_FP32_POLICY
    reports = _paired_score_reports((6e-6, 6e-6))
    with pytest.raises(AssertionError, match='mean signed drift'):
        assert_pereira_score_drift(reports, policy=L4_FP32_POLICY)
