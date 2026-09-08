"""Compare both unified model routes with a separately run legacy reference.

This module never downloads weights. The runner is opt-in and uses explicit
local checkpoints; benchmark assemblies/ceilings/stimuli must already be staged.
"""
import gc
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParityCase:
    legacy: str
    domain: str

    @property
    def unified(self):
        return self.legacy + '-unified'


CASES = tuple(
    ParityCase(f'MajajHong2015{access}.{region}-pls', 'vision')
    for access in ('', 'public') for region in ('V4', 'IT')
) + tuple(ParityCase(f'Pereira2018.{experiment}sentences-linear', 'language')
          for experiment in (243, 384))


PEREIRA_CASES = tuple(case for case in CASES if case.domain == 'language')
ADAPTER_SCORE_ULPS = 4

# Fixed GPT-2/Pereira regression policy, not a universal forward-error theorem.
# User GPU evidence (A10G, torch 2.6.0, transformers 4.57.6, FP32, matmul TF32 off):
# all 243/384 token, position and mask audits pass; no cache sliding. Default
# h.11 deltas are exactly 2^-13 at rounded references 273.237/278.512. Both references
# lie in [2^8, 2^9), so spacing = 2^-15 and 2^-13 = 4 * 2^-15 (four ULPs).
# IEEE eager/math still differ: sampled worst-absolute-error coordinates give
# 4/11/12 ULPs, and math gives 665 ULPs at 1.619 (7.927417755126953e-5).
# The other FP32 ablations: 243 eager 8.392333984375e-5 at 99.0666 (11 ULPs);
# 384 math 9.1552734375e-5 at 125.896 (12 ULPs), eager 6.103515625e-5 at
# -134.127 (4 ULPs). These are absolute-error maxima, not maximum ULP ratios.
# 1.619 is not numerically near zero; it is small relative to this residual
# stream. Local output ULPs alone cannot bound accumulated/cancellation error.
# FP64 eager collapses the worst discrepancies to 2.5579538487363607e-13 and
# 3.126388037344441e-13. This supports FP32 execution sensitivity; changing the
# FP32 attention implementation alone does not remove it. Kernel choice still
# affects the size. Max-reference magnitudes are 351.91/335.72; these summaries
# do not identify a particular "rogue dimension" or the largest ULP ratio.
#
# Choose a four-fraction-bit error budget: 16 effective FP32 ULPs, the smallest
# power of two above the reported 12. This retains roughly 19 fraction bits at
# the effective reference scale. A magnitude floor of 64 is the smallest binary
# scale making the 665-local-ULP small-coordinate example fit this budget: it is
# 10.390625 ULPs at 64. This is an explicit empirical calibration, not a derived
# bound on arbitrary inputs. No element is excluded. Limits are 2^-13 below 64
# and 2^-11 in [256, 512); errors above each element's limit fail. The floor is
# intentionally a mixed absolute/ULP policy and needs the score/bias guards below.
PEREIRA_NATIVE_ULPS = 16
PEREIRA_ULP_MAGNITUDE_FLOOR = 64.0

# Published-score agreement budget: 0.002 = 0.2 percentage points of the [0, 1]
# ceiling-normalized scale, not an estimated noise variance or scientific law.
# Observed native-minus-legacy deltas: +0.0012075814903310 (243) and
# -0.0003588701211292 (384), raw +0.00042722938546023/-0.00013042685277098.
# The budget is <2x the largest normalized observation, rather than 0.01 (>8x).
# Raw deltas must meet the same budget after dividing by the common ceiling,
# so clipping cannot conceal regression. Repeats and native 512/1024 are exact.
PEREIRA_SCORE_ATOL = 0.002

# A loose unsigned per-case bound cannot detect common-sign bias or growing
# drift within that bound. The paired slow test therefore also rejects two
# same-sign nonzero deltas and departures beyond the rounding intervals of the
# supplied four-decimal magnitudes 0.0012/0.0004 (half a last digit = 0.00005).
# These frozen historical limits catch growth to 0.0013/0.0005 even though both
# pass 0.002. Sub-resolution changes within these intervals remain possible;
# no finite tolerance can detect every infinitesimal or score-invariant defect.
PEREIRA_SCORE_DRIFT_LIMITS = {
    PEREIRA_CASES[0].unified: 0.00125,
    PEREIRA_CASES[1].unified: 0.00045,
}


def uses_pereira_policy(case):
    return case in PEREIRA_CASES


def fp32_ulp_tolerance(reference):
    """Elementwise 16-ULP limits with the declared reference-magnitude floor."""
    reference = np.asarray(reference, dtype=np.float64)
    assert np.isfinite(reference).all(), 'Non-finite reference activations'
    assert (np.abs(reference) <= np.finfo(np.float32).max).all(), 'Reference exceeds FP32 range'
    magnitude = np.maximum(np.abs(reference.astype(np.float32)), PEREIRA_ULP_MAGNITUDE_FLOOR)
    # frexp's exponent is one larger than the binade exponent: FP32 spacing
    # is 2^(exponent-24). Float64 ldexp avoids nextafter(max_float32) -> inf.
    exponent = np.frexp(magnitude)[1]
    return PEREIRA_NATIVE_ULPS * np.ldexp(np.ones(reference.shape), exponent - 24)


def assert_pereira_activations(actual, reference, *, err_msg='native activations differ'):
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape, f'{err_msg}: activation shapes differ'
    assert np.isfinite(actual).all(), f'{err_msg}: non-finite activations'
    limits = fp32_ulp_tolerance(reference)
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    ratios = delta / (limits / PEREIRA_NATIVE_ULPS)
    maximum = float(np.max(ratios))
    assert (delta <= limits).all(), (
        f'{err_msg}: maximum {maximum:.9g} effective FP32 ULPs exceeds '
        f'{PEREIRA_NATIVE_ULPS} (magnitude floor {PEREIRA_ULP_MAGNITUDE_FLOOR:g})')
    return maximum


def assert_pereira_score_drift(reports):
    """Joint guard; callers must supply both fixed Pereira benchmark results."""
    pairs = [report for report in reports if report['benchmark'] in PEREIRA_SCORE_DRIFT_LIMITS]
    assert len(pairs) == 2 and {p['benchmark'] for p in pairs} == set(PEREIRA_SCORE_DRIFT_LIMITS), \
        'Paired Pereira score guard requires both benchmarks exactly once'
    for key in ('score', 'raw'):
        deltas = []
        for report in pairs:
            reference, native = report['routes']['legacy'], report['routes']['native']
            delta = _scalar(native[key]) - _scalar(reference[key])
            if key == 'raw':
                ceiling = _scalar(reference['ceiling'])
                assert ceiling > 0, 'Pereira ceiling must be positive'
                delta /= ceiling
            deltas.append(delta)
            limit = PEREIRA_SCORE_DRIFT_LIMITS[report['benchmark']]
            assert abs(delta) <= limit, (
                f"{report['benchmark']}: native {key} drift {delta:.17g} exceeds "
                f'historical rounding envelope {limit:g}')
        assert not (all(delta > 0 for delta in deltas) or all(delta < 0 for delta in deltas)), \
            f'Pereira native {key}: same-sign drift across both benchmarks: {deltas}'


def _scalar(value):
    array = np.asarray(value)
    if array.size != 1 or not np.isfinite(array).all():
        raise AssertionError(f'Expected a finite scalar score, got {array}')
    return float(array.reshape(-1)[0])


def adapter_score_tolerance(reference):
    """Four FP32 spacings at the scalar's magnitude; no absolute-error floor."""
    magnitude = abs(_scalar(reference))
    assert magnitude <= np.finfo(np.float32).max, 'Adapter reference score exceeds FP32 range'
    magnitude = np.float32(magnitude)
    exponent = np.frexp(magnitude)[1]
    # Clamp at the subnormal spacing, including zero. ldexp avoids infinity
    # from nextafter(max_float32); no rounding of either score before comparison.
    spacing = np.ldexp(1., max(int(exponent) - 24, -149)) if magnitude else 2. ** -149
    return ADAPTER_SCORE_ULPS * spacing


def validate_case(case, candidate_factory, benchmark_loader=None, *,
                  native_atol=1e-6, score_atol=None):
    """Run three independent conditions, retaining inputs as well as scores.

    ``candidate_factory(case, route)`` must return fresh, matched-weight models
    for ``legacy``, ``adapter``, and ``native``. Native must be a BrainScoreModel;
    an adapter passed as native is rejected. Fixed recording layers avoid layer
    search confounds. Real score execution belongs exclusively in the slow tier.
    Fixed Pereira cases use the ULP policy above; ``native_atol`` applies only
    to other cases. Call ``assert_pereira_score_drift`` on both Pereira reports
    as well: per-case validation alone cannot detect joint score bias.
    """
    from brainscore import load_benchmark
    from brainscore_core.model_interface import BrainScoreModel
    from brainscore_language.compat.unified_adapter import LanguageModelAdapter
    from brainscore_vision.compat.unified_adapter import VisionModelAdapter
    if os.environ.get('RESULTCACHING_DISABLE') != '1':
        raise RuntimeError('Parity validation requires RESULTCACHING_DISABLE=1')
    benchmark_loader = benchmark_loader or load_benchmark
    pereira = uses_pereira_policy(case)
    if score_atol is None:
        score_atol = PEREIRA_SCORE_ATOL if pereira else 1e-6
    adapters = (VisionModelAdapter, LanguageModelAdapter)
    runs = {}
    reference = None
    for route in ('legacy', 'adapter', 'native'):
        candidate = candidate_factory(case, route)
        if route == 'native' and not isinstance(candidate, BrainScoreModel):
            raise TypeError('native route requires BrainScoreModel, not an adapter')
        if route == 'adapter' and not isinstance(candidate, adapters):
            raise TypeError('adapter route requires a domain adapter')
        if route == 'legacy' and isinstance(candidate, (*adapters, BrainScoreModel)):
            raise TypeError('legacy route requires the original domain subject')
        benchmark = benchmark_loader(case.legacy if route == 'legacy' else case.unified)
        attr = '_similarity_metric' if case.domain == 'vision' else 'metric'
        metric = getattr(benchmark, attr)
        captured = []
        def capture(source, target, *args, **kwargs):
            source = source.transpose('presentation', 'neuroid')
            # Model names in neuroid_id legitimately differ between routes.
            captured.append((np.array(source.stimulus_id.values, copy=True),
                             np.array(source.values, copy=True),
                             np.array(target.stimulus_id.values, copy=True)))
            return metric(source, target, *args, **kwargs)
        setattr(benchmark, attr, capture)
        try:
            score = benchmark(candidate)
            observed = {'score': _scalar(score), 'raw': _scalar(score.attrs['raw'])}
            if pereira:
                observed['ceiling'] = _scalar(benchmark.ceiling)
                assert observed['ceiling'] > 0, 'Pereira ceiling must be positive'
                if route != 'legacy':
                    np.testing.assert_equal(observed['ceiling'], runs['legacy']['ceiling'])
            if not captured:
                raise AssertionError('Benchmark did not call its metric')
            if route == 'legacy':
                reference = captured
                observed['max_activation_delta'] = 0.0
            else:
                assert len(captured) == len(reference), 'metric call count differs'
                deltas, activation_errors, ulps = [], [], []
                for expected, actual in zip(reference, captured):
                    ids, values, target_ids = actual
                    np.testing.assert_array_equal(ids, expected[0], err_msg=f'{route}: stimulus order')
                    np.testing.assert_array_equal(target_ids, expected[2], err_msg=f'{route}: target order')
                    try:
                        if route == 'native' and pereira:
                            # Use the calibrated policy and independent score
                            # gates above; never apply these allowances to adapters.
                            ulps.append(assert_pereira_activations(values, expected[1],
                                err_msg=f'{case.unified}: native activations differ'))
                        else:
                            np.testing.assert_allclose(values, expected[1], rtol=0,
                                atol=0 if route == 'adapter' else native_atol,
                                err_msg=f'{case.unified}: {route} activations differ')
                    except AssertionError as error:
                        if route == 'adapter':
                            raise  # Exact compatibility contract on every benchmark.
                        activation_errors.append(str(error))
                    deltas.append(float(np.max(np.abs(values - expected[1]))))
                observed['max_activation_delta'] = max(deltas)
                if ulps:
                    observed['max_effective_fp32_ulps'] = max(ulps)
                # Keep raw and published scores visible even if activations fail.
                # The rationale and measured values are beside the policy above.
                score_errors = []
                for key in ('score', 'raw'):
                    try:
                        if route == 'adapter':
                            # Ubuntu CPU CI: delta 3.7252903e-9 at ~0.03181,
                            # relative 1.17111432e-7: one FP32 ULP (2^-28).
                            # The synthetic metric is mean(abs(FP32 values)),
                            # not regression/BLAS. Equal Pereira values arrive
                            # with different C/F layouts; NumPy reduction order
                            # can change. Vision uses the same synthetic metric;
                            # its real PLS/correlation metric also reduces floats.
                            # Allow four ULPs (two fraction bits) for scalar
                            # reduction/normalization portability: 1.49011612e-8
                            # at 0.03181, scaling with each raw/published score.
                            # This is a small acceptance budget, not a bound on
                            # arbitrary regression conditioning. Crucially, the
                            # adapter's activation check above stays atol=rtol=0:
                            # any changed activation fails before this allowance
                            # can mask it. Score-only changes beyond four ULPs
                            # still fail; native tolerances cannot enlarge this.
                            limit = adapter_score_tolerance(runs['legacy'][key])
                        else:
                            limit = score_atol
                        if route != 'adapter' and pereira and key == 'raw':
                            limit *= runs['legacy']['ceiling']
                        np.testing.assert_allclose(observed[key], runs['legacy'][key], rtol=0,
                            atol=limit,
                            err_msg=f'{case.unified}: {route} {key} differs')
                    except AssertionError as error:
                        if route == 'adapter':
                            raise
                        score_errors.append(str(error))
                if score_errors or activation_errors:
                    raise AssertionError('\n\n'.join(score_errors + activation_errors))
            runs[route] = observed
        finally:
            setattr(benchmark, attr, metric)
            if hasattr(candidate, 'reset'):
                candidate.reset()
            del candidate, benchmark
            gc.collect()
    return {'benchmark': case.unified, 'native_atol': None if pereira else native_atol,
            'native_ulps': PEREIRA_NATIVE_ULPS if pereira else None,
            'native_ulp_magnitude_floor': PEREIRA_ULP_MAGNITUDE_FLOOR if pereira else None,
            'score_atol': score_atol, 'routes': runs}


def local_candidate(case, route):
    """Reference pairs: local ResNet18 with fixed V4/IT layers, or local GPT-2.

    UMI_PARITY_RESNET18 is a torchvision ResNet18 state_dict file.
    UMI_PARITY_GPT2 is a local Hugging Face GPT-2 model/tokenizer directory.
    These models test interface parity; they are not scientific layer mappings.
    """
    from brainscore_core.model_interface import BrainScoreModel
    if case.domain == 'vision':
        import functools
        import torch
        from torchvision.models import resnet18
        from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
        from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
        from brainscore_vision.compat.unified_adapter import VisionModelAdapter
        net = resnet18(weights=None)
        net.load_state_dict(torch.load(os.environ['UMI_PARITY_RESNET18'], map_location='cpu', weights_only=True))
        net.eval()
        mapping = {'V4': 'layer2', 'IT': 'layer4'}
        wrapper = PytorchWrapper(net, functools.partial(load_preprocess_images, image_size=224),
                                 identifier='parity-resnet18', batch_size=4)
        if route == 'native':
            return BrainScoreModel('parity-native', model=net, region_layer_map=mapping,
                preprocessors={'vision': wrapper}, visual_degrees=8)
        legacy = ModelCommitment('parity-legacy', wrapper, layers=list(mapping.values()),
                                 region_layer_map=mapping, visual_degrees=8)
        return VisionModelAdapter(legacy) if route == 'adapter' else legacy
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from brainscore.model_helpers.text_wrapper import TextWrapper
    from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
    from brainscore_language.compat.unified_adapter import LanguageModelAdapter
    path = os.environ['UMI_PARITY_GPT2']
    net = AutoModelForCausalLM.from_pretrained(path, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    mapping = {'language_system': 'transformer.h.11'}
    if route == 'native':
        wrapper = TextWrapper(net, tokenizer, identifier='parity-native', max_length=1024, batch_size=1)
        return BrainScoreModel('parity-native', model=net, region_layer_map=mapping,
                               preprocessors={'text': wrapper})
    legacy = HuggingfaceSubject('parity-legacy', mapping, model=net, tokenizer=tokenizer)
    return LanguageModelAdapter(legacy) if route == 'adapter' else legacy
