"""Offline checks for the investigation tool, not benchmark scoring tests."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from brainscore.validation.language_parity_diagnostics import difference, gpu_run, tiny_cpu
from tests.test_umi_review_regressions import language_pair

pytestmark = pytest.mark.integration


def test_difference_reports_the_values_and_spacing_behind_power_of_two():
    expected = np.array([[1536.]], dtype=np.float32)
    actual = np.nextafter(expected, np.float32(np.inf))
    report = difference(expected, actual)
    assert report['max_abs'] == 2 ** -13
    assert report['max_abs_hex'] == (2 ** -13).hex()
    assert report['worst'][0]['delta_in_reference_spacings'] == 1
    assert report['worst'][0]['reference'] == 1536
    assert not report['passes']
    with pytest.raises(ValueError, match='Non-finite'):
        difference(expected, np.array([[np.nan]]))


def test_tiny_wrapper_differences_survive_bypassing_wrappers():
    threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            report = tiny_cpu()
    finally:
        torch.set_num_threads(threads)
    for name, result in report.items():
        if name == 'fp32_spacing':
            continue
        for row in result['token_audit']:
            for key in ('prefix_ids_equal', 'cache_not_slid', 'legacy_mask_all_ones',
                        'native_mask_all_ones', 'native_positions_zero_based',
                        'legacy_positions_equal', 'legacy_causal_mask_correct',
                        'native_causal_mask_correct'):
                assert row[key], (name, row)
        for key in ('native_vs_direct_full', 'legacy_vs_direct_cached'):
            assert all(delta['max_abs'] == 0 for delta in result[key].values())
        # FP64 is captured before wrapper conversion to FP32.
        if 'float64' in name:
            assert result['parameter_dtypes'] == ['torch.float64']
            assert result['cached_vs_full']['transformer.h.3']['max_abs'] < 1e-12


def test_diagnostic_saves_scores_even_when_activation_parity_fails(monkeypatch, tmp_path):
    import brainscore.validation.language_parity_diagnostics as diagnostic
    from brainscore.validation import benchmark_parity

    monkeypatch.setenv('RUN_UMI_PARITY', '1')
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    monkeypatch.setenv('UMI_PARITY_GPT2', str(tmp_path))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(diagnostic, 'environment', lambda: {'synthetic': True})
    monkeypatch.setattr(benchmark_parity, 'CASES', benchmark_parity.CASES[-1:])
    observed = []
    def observe(case, route, max_length, score, score_float64):
        observed.append((route, max_length, score))
        native = route == 'native'
        value = np.array([[1. + 0.01 * native, 2.]], dtype=np.float32)
        calls = [dict(input_ids=[[2]], attention_mask=[[1]], past_length=0,
                      effective_position_ids=[[0]], internal_attention_mask=None)]
        meta = {'score': 0.8 - 0.003 * native, 'raw': 0.4 - 0.0015 * native, 'ceiling': 0.5}
        trace = SimpleNamespace(calls=calls, values={'transformer.h.11': value})
        return meta, value, np.array(['s0']), np.array(['s0']), trace
    monkeypatch.setattr(diagnostic, 'observe_case', observe)
    gpu_run(SimpleNamespace(repeats=2, score=True, score_float64=False,
                            ablations=False, output=tmp_path))
    report = json.loads((tmp_path / 'report.json').read_text())
    case = next(iter(report['cases'].values()))
    assert len(observed) == 8
    assert len(list(tmp_path.glob('*.npz'))) == 8
    native = case['comparisons']['legacy_0_vs_native_1024_0']
    assert not native['activations']['passes']
    assert not native['score_passes']
    assert native['score_delta'] == pytest.approx(-0.003)
    assert native['activations']['policy'] == {'ulps': 16, 'magnitude_floor': 64.0}
    assert native['score_atol'] == 0.002
    assert native['raw_atol'] == 0.001
    assert case['runs']['native_512_1']['score'] == pytest.approx(0.797)
    assert case['max_length_score_delta']['score'] == 0
    assert case['score_repeat_ranges']['native_1024']['score'] == 0


def test_observe_actual_tiny_routes_and_replay_ablations(monkeypatch, language_pair):
    import brainscore
    from brainscore.validation import benchmark_parity
    from brainscore_core.metrics import Score
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    from brainscore.validation.language_parity_diagnostics import ablate, observe_case

    native, adapter = language_pair
    native._preprocessors['text']._batch_size = 1
    def candidate_factory(case, route):
        native.reset()
        adapter.reset()
        return {'native': native, 'adapter': adapter, 'legacy': adapter._legacy}[route]
    monkeypatch.setattr(benchmark_parity, 'local_candidate', candidate_factory)

    class SyntheticMetric:
        def __init__(self):
            self.regression = SimpleNamespace(fit=lambda source, target: None,
                _regression=SimpleNamespace(rank_=1, singular_=np.array([1.])))

        def __call__(self, source, target):
            self.regression.fit(source, target)
            return Score(0.25)

    class SyntheticBenchmark:
        def __init__(self, identifier):
            self.unified = identifier.endswith('-unified')
            self.metric, self.ceiling = SyntheticMetric(), Score(0.5)
            self.data = NeuroidAssembly(np.zeros((2, 1)), dims=['presentation', 'neuroid'],
                coords={'stimulus': ('presentation', ['the cat', 'cat sat']),
                        'stimulus_id': ('presentation', ['s0', 's1']),
                        'passage_label': ('presentation', ['p', 'p']),
                        'neuroid_id': ('neuroid', ['n0'])})

        def __call__(self, candidate):
            if self.unified:
                candidate.start_recording('language_system', recording_type='fMRI')
                stimuli = StimulusSet(dict(sentence=list(self.data.stimulus.values),
                    stimulus_id=list(self.data.stimulus_id.values), context_id=['p', 'p']))
                source = candidate.process(stimuli)
            else:
                candidate.start_neural_recording('language_system', 'fMRI')
                source = candidate.digest_text(list(self.data.stimulus.values))['neural']
                source = source.assign_coords(stimulus_id=('presentation', ['s0', 's1']))
            raw = self.metric(source, self.data)
            value = Score(float(raw / self.ceiling))
            value.attrs['raw'] = raw
            return value

    monkeypatch.setattr(brainscore, 'load_benchmark', SyntheticBenchmark)
    case = benchmark_parity.CASES[-1]
    traces = {}
    for route in ('legacy', 'adapter', 'native'):
        meta, values, ids, target_ids, traces[route] = observe_case(
            case, route, score=True, score_float64=True)
        assert meta['score'] == meta['float64_regression']['score'] == 0.5
        assert meta['regression_fits'][0]['dtype'] == 'float32'
        assert meta['float64_regression']['regression_fits'][0]['dtype'] == 'float64'
        assert meta['hook_vs_metric_input']['max_abs'] == 0
        assert meta['untruncated_lengths'] == [2, 4]
        np.testing.assert_array_equal(ids, target_ids)
    original_precision = torch.get_float32_matmul_precision()
    ablations = ablate(case, traces['legacy'], traces['native'], worst_row=1)
    assert torch.get_float32_matmul_precision() == original_precision
    assert set(ablations) == {'default', 'ieee_sdpa_math', 'ieee_eager', 'float64_eager'}
    assert ablations['default']['baseline_full_reproduced']['max_abs'] == 0
    assert ablations['default']['baseline_cached_reproduced']['max_abs'] == 0
