"""Compare the adapter and native routes with a separately run legacy reference.

Each route scores the same legacy benchmark: the legacy route calls it with the original
domain subject; the adapter and native routes reach it through the legacy-benchmark facade,
as ``brainscore.score`` does (``subject_for_benchmark``).

This module never downloads weights. The runner is opt-in and uses explicit
local checkpoints; benchmark assemblies/ceilings/stimuli must already be staged.
"""
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .parity_storage import CHUNK_BYTES, snapshot


@dataclass(frozen=True)
class ParityCase:
    legacy: str
    domain: str


CASES = tuple(
    ParityCase(f'MajajHong2015{access}.{region}-pls', 'vision')
    for access in ('', 'public') for region in ('V4', 'IT')
)


RIDGE_CASES = tuple(ParityCase(f'Pereira2018.{experiment}sentences-ridge', 'language')
                    for experiment in (243, 384))
# Ridge uses strict defaults until a separately reviewed calibration exists.
RELEASE_CASES = CASES + RIDGE_CASES


# Every budget below (ULPs, score tolerances, drift limits) was measured when the adapter and
# native routes scored the `-unified` copies of these benchmarks with the typed calls
# (start_recording, process). The routes now score the legacy benchmarks through the facade;
# the values are kept unchanged until a cloud run of run_parity on the release cases confirms
# them. No value changes without a measurement.
ADAPTER_SCORE_ULPS = 4
HISTORICAL_POLICY = 'historical-v1'
CPU_FP32_POLICY = 'gpt2-pereira-cpu-fp32-v1'
L4_FP32_POLICY = 'gpt2-pereira-l4-fp32-v1'
FP32_POLICIES = (CPU_FP32_POLICY, L4_FP32_POLICY)
POLICIES = (HISTORICAL_POLICY, *FP32_POLICIES)
# Scoped to the two GPT-2/Pereira ridge cases, not arbitrary language inputs.
# See docs/archive/numerical_policy.md for the precision decision and counterexamples.
CPU_NATIVE_ULPS = 32
# The L4 profile keeps the 16-ULP activation budget calibrated below.
L4_NATIVE_ULPS = 16
CPU_SCORE_ATOL = 1e-5
CPU_MEAN_DRIFT_ATOL = 5e-6


class ParityFailure(AssertionError):
    """A failed comparison with the observations needed to diagnose it."""
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report

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
# Choose a four-fraction-bit error budget: 16 effective FP32 ULPs (L4_NATIVE_ULPS), the smallest
# power of two above the reported 12. This retains roughly 19 fraction bits at
# the effective reference scale. A magnitude floor of 64 is the smallest binary
# scale making the 665-local-ULP small-coordinate example fit this budget: it is
# 10.390625 ULPs at 64. This is an explicit empirical calibration, not a derived
# bound on arbitrary inputs. No element is excluded. Limits are 2^-13 below 64
# and 2^-11 in [256, 512); errors above each element's limit fail. The floor is
# intentionally a mixed absolute/ULP policy and needs the score/bias guards below.
PEREIRA_ULP_MAGNITUDE_FLOOR = 64.0


def fp32_ulp_tolerance(reference, *, ulps):
    """Elementwise ULP limits with the declared reference-magnitude floor."""
    reference = np.asarray(reference, dtype=np.float64)
    assert np.isfinite(reference).all(), 'Non-finite reference activations'
    assert (np.abs(reference) <= np.finfo(np.float32).max).all(), 'Reference exceeds FP32 range'
    magnitude = np.maximum(np.abs(reference.astype(np.float32)), PEREIRA_ULP_MAGNITUDE_FLOOR)
    # frexp's exponent is one larger than the binade exponent: FP32 spacing
    # is 2^(exponent-24). Float64 ldexp avoids nextafter(max_float32) -> inf.
    exponent = np.frexp(magnitude)[1]
    return ulps * np.ldexp(np.ones(reference.shape), exponent - 24)


def assert_pereira_activations(actual, reference, *, ulps, err_msg='native activations differ'):
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape, f'{err_msg}: activation shapes differ'
    assert np.isfinite(actual).all(), f'{err_msg}: non-finite activations'
    limits = fp32_ulp_tolerance(reference, ulps=ulps)
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    ratios = delta / (limits / ulps)
    maximum = float(np.max(ratios))
    assert (delta <= limits).all(), (
        f'{err_msg}: maximum {maximum:.9g} effective FP32 ULPs exceeds '
        f'{ulps} (magnitude floor {PEREIRA_ULP_MAGNITUDE_FLOOR:g})')
    return maximum


def assert_pereira_score_drift(reports, *, policy):
    """Joint guard of an FP32 policy; callers must supply both Pereira ridge results."""
    if policy not in FP32_POLICIES:
        raise ValueError(f'The paired Pereira guard needs an FP32 policy {FP32_POLICIES}; got {policy}')
    names = {case.legacy for case in RIDGE_CASES}
    pair = [report for report in reports if report['benchmark'] in names]
    assert len(pair) == 2 and {p['benchmark'] for p in pair} == names, \
        'FP32 Pereira score guard requires both experiments'
    for key in ('score', 'raw'):
        deltas = []
        for report in pair:
            reference, native = report['routes']['legacy'], report['routes']['native']
            delta = _scalar(native[key]) - _scalar(reference[key])
            if key == 'raw':
                ceiling = _scalar(reference['ceiling'])
                assert ceiling > 0, 'Pereira ceiling must be positive'
                delta /= ceiling
            assert abs(delta) <= CPU_SCORE_ATOL, f'{policy}: {key} drift exceeds per-case budget'
            deltas.append(delta)
        assert abs(sum(deltas) / len(deltas)) <= CPU_MEAN_DRIFT_ATOL, \
            f'{policy}: {key}: mean signed drift exceeds budget: {deltas}'


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


def validate_case(case, subject_factory, benchmark_loader=None, *,
                  native_atol=1e-6, score_atol=None, policy=HISTORICAL_POLICY,
                  scratch_dir=None):
    """Run three independent conditions, retaining inputs as well as scores.

    ``subject_factory(case, route)`` must return fresh, matched-weight models
    for ``legacy``, ``adapter``, and ``native``. Native must be a BrainScoreModel;
    an adapter passed as native is rejected. Fixed recording layers avoid layer
    search confounds. Real score execution belongs exclusively in the slow tier.
    Activation snapshots use temporary files; comparisons read bounded chunks.
    ``scratch_dir`` selects their parent directory; files are removed on exit.
    The Pereira ridge cases use the ULP policy above under an FP32 policy;
    ``native_atol`` applies to the other cases. Under an FP32 policy, call
    ``assert_pereira_score_drift`` on both ridge reports as well: per-case
    validation alone cannot detect joint score bias.
    """
    with TemporaryDirectory(prefix='umi-parity-', dir=scratch_dir) as directory:
        return _validate_case(
            case, subject_factory, benchmark_loader,
            native_atol=native_atol, score_atol=score_atol, policy=policy,
            directory=Path(directory),
        )


def _compare_snapshots(actual, reference, *, route, case, native_atol, pereira, native_ulps):
    assert actual.shape == reference.shape, f'{case.legacy}: {route} activation shapes differ'
    assert np.prod(actual.shape) > 0, f'{case.legacy}: empty activations'
    # Both files use the same element boundaries even if their dtypes differ.
    elements = CHUNK_BYTES // max(actual.dtype.itemsize, reference.dtype.itemsize)
    max_delta, max_ulps = 0., 0.
    error = None
    for values, expected in zip(actual.chunks(elements), reference.chunks(elements)):
        try:
            if route == 'native' and pereira:
                max_ulps = max(max_ulps, assert_pereira_activations(
                    values, expected, err_msg=f'{case.legacy}: native activations differ',
                    ulps=native_ulps))
            else:
                np.testing.assert_allclose(
                    values, expected, rtol=0, atol=0 if route == 'adapter' else native_atol,
                    err_msg=f'{case.legacy}: {route} activations differ',
                )
        except AssertionError as failure:
            if route == 'adapter':
                raise
            # Retain one useful error without accumulating a message per chunk.
            error = error or str(failure)
        max_delta = float(np.maximum(max_delta, np.max(np.abs(values - expected))))
    return max_delta, max_ulps, error


def _validate_case(case, subject_factory, benchmark_loader, *,
                   native_atol, score_atol, policy, directory):
    from brainscore import load_benchmark
    from brainscore import BrainScoreModel
    from brainscore_language.compat.unified_adapter import LanguageModelAdapter
    from brainscore_vision.compat.unified_adapter import VisionModelAdapter
    from brainscore_core.legacy_facade import declared_visual_degrees, subject_for_benchmark
    if os.environ.get('RESULTCACHING_DISABLE') != '1':
        raise RuntimeError('Parity validation requires RESULTCACHING_DISABLE=1')
    benchmark_loader = benchmark_loader or load_benchmark
    if policy not in POLICIES:
        raise ValueError(f'Unknown numerical policy: {policy}')
    fixed_pereira = policy in FP32_POLICIES and case in RIDGE_CASES
    pereira = fixed_pereira
    native_ulps = {CPU_FP32_POLICY: CPU_NATIVE_ULPS, L4_FP32_POLICY: L4_NATIVE_ULPS}.get(policy)
    if fixed_pereira and score_atol is not None:
        raise ValueError('The versioned FP32 policy has a fixed score budget')
    if score_atol is None:
        score_atol = CPU_SCORE_ATOL if fixed_pereira else 1e-6
    adapters = (VisionModelAdapter, LanguageModelAdapter)
    runs = {}
    reference = None
    def report():
        return {'benchmark': case.legacy, 'policy': policy,
                'native_atol': None if pereira else native_atol,
                'native_ulps': native_ulps if pereira else None,
                'native_ulp_magnitude_floor': PEREIRA_ULP_MAGNITUDE_FLOOR if pereira else None,
                'score_atol': score_atol, 'routes': runs}
    for route in ('legacy', 'adapter', 'native'):
        subject = subject_factory(case, route)
        if route == 'native' and not isinstance(subject, BrainScoreModel):
            raise TypeError('native route requires BrainScoreModel, not an adapter')
        if route == 'adapter' and not isinstance(subject, adapters):
            raise TypeError('adapter route requires a domain adapter')
        if route == 'legacy' and isinstance(subject, (*adapters, BrainScoreModel)):
            raise TypeError('legacy route requires the original domain subject')
        benchmark = benchmark_loader(case.legacy)
        attr = '_similarity_metric' if case.domain == 'vision' else 'metric'
        metric = getattr(benchmark, attr)
        captured = []
        def capture(source, target, *args, **kwargs):
            source = source.transpose('presentation', 'neuroid')
            # Model names in neuroid_id legitimately differ between routes.
            captured.append((np.array(source.stimulus_id.values, copy=True),
                             snapshot(source.values, directory / f'{route}-{len(captured)}.bin'),
                             np.array(target.stimulus_id.values, copy=True)))
            return metric(source, target, *args, **kwargs)
        setattr(benchmark, attr, capture)
        try:
            # The legacy route calls the benchmark with the domain subject itself; the
            # adapter and native routes receive what brainscore.score gives them (the facade).
            received = subject if route == 'legacy' else subject_for_benchmark(
                subject, benchmark, visual_degrees=declared_visual_degrees(subject))
            score = benchmark(received)
            observed = {'score': _scalar(score), 'raw': _scalar(score.attrs['raw'])}
            runs[route] = observed
            observed['metric_inputs'] = [
                {'shape': list(values.shape), 'dtype': str(values.dtype),
                 'values_sha256': values.sha256}
                for _, values, _ in captured]
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
                    delta, effective_ulps, error = _compare_snapshots(
                        values, expected[1], route=route, case=case,
                        native_atol=native_atol, pereira=pereira, native_ulps=native_ulps,
                    )
                    deltas.append(delta)
                    if error:
                        activation_errors.append(error)
                    elif route == 'native' and pereira:
                        ulps.append(effective_ulps)
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
                            err_msg=f'{case.legacy}: {route} {key} differs')
                    except AssertionError as error:
                        if route == 'adapter':
                            raise
                        score_errors.append(str(error))
                if score_errors or activation_errors:
                    raise AssertionError('\n\n'.join(score_errors + activation_errors))
            runs[route] = observed
        except AssertionError as error:
            raise ParityFailure(str(error), report()) from error
        finally:
            if route != 'legacy':
                for _, values, _ in captured:
                    values.path.unlink(missing_ok=True)
            setattr(benchmark, attr, metric)
            pending = sys.exc_info()[1]
            try:
                # A session subject declares reset; only an original old model may lack it.
                reset = getattr(subject, 'reset', None) if route == 'legacy' else subject.reset
                if reset is not None:
                    reset()
            except Exception as reset_error:
                if pending is None:
                    raise
                pending.add_note(f'The reset of the subject also failed: {reset_error!r}')
            finally:
                del subject, benchmark
                gc.collect()
    return report()


def local_subject(case, route, *, language_precision='float32', execution_observer=None):
    """Reference pairs: local ResNet18 with fixed V4/IT layers, or local GPT-2.

    UMI_PARITY_RESNET18 is a torchvision ResNet18 state_dict file.
    UMI_PARITY_GPT2 is a local Hugging Face GPT-2 model/tokenizer directory.
    These models test interface parity; they are not scientific layer mappings.
    """
    from brainscore import BrainScoreModel
    def finish(subject, net):
        if execution_observer is not None:
            execution_observer({
                'parameter_devices': sorted({str(p.device) for p in net.parameters()}),
                'parameter_dtypes': sorted({str(p.dtype) for p in net.parameters()}),
                'attention_implementation': getattr(getattr(net, 'config', None), '_attn_implementation', None),
            })
        return subject
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
            return finish(BrainScoreModel('parity-native', model=net, region_layer_map=mapping,
                extractor={'vision': wrapper}, visual_degrees=8), net)
        legacy = ModelCommitment('parity-legacy', wrapper, layers=list(mapping.values()),
                                 region_layer_map=mapping, visual_degrees=8)
        return finish(VisionModelAdapter(legacy) if route == 'adapter' else legacy, net)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from brainscore.model_helpers.text_wrapper import TextWrapper
    from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
    from brainscore_language.compat.unified_adapter import LanguageModelAdapter
    path = os.environ['UMI_PARITY_GPT2']
    net = AutoModelForCausalLM.from_pretrained(path, local_files_only=True).eval()
    if language_precision == 'float64-eager':
        net.double()
        net.set_attn_implementation('eager')
    elif language_precision != 'float32':
        raise ValueError(f'Unsupported language precision: {language_precision}')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    mapping = {'language_system': 'transformer.h.11'}
    if route == 'native':
        wrapper = TextWrapper(net, tokenizer, identifier='parity-native', max_length=1024, batch_size=1)
        return finish(BrainScoreModel('parity-native', model=net, region_layer_map=mapping,
                               extractor={'text': wrapper}), net)
    legacy = HuggingfaceSubject('parity-legacy', mapping, model=net, tokenizer=tokenizer)
    return finish(LanguageModelAdapter(legacy) if route == 'adapter' else legacy, net)
