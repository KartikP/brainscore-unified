"""Opt-in Pereira parity diagnostics; never changes acceptance tolerances.

--tiny-cpu uses random, small GPT-2 models and a synthetic tokenizer only.
--gpu requires RUN_UMI_PARITY=1 and UMI_PARITY_GPT2 pointing to staged files.
Reports retain strict absolute diagnostics and apply the shared Pereira ULP
and score policy. Reports are written even when acceptance criteria fail.
"""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


def difference(reference, actual, atol=1e-6):
    """Report exact differences and local FP32 spacing, not rounded maxima."""
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape:
        raise ValueError(f'Activation shapes differ: {reference.shape}, {actual.shape}')
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('Non-finite activations')
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    maximum = float(delta.max())
    worst = []
    for flat in np.flatnonzero(delta == maximum)[:8]:
        index = np.unravel_index(flat, delta.shape)
        expected, observed = float(reference[index]), float(actual[index])
        spacing = float(np.spacing(np.float32(abs(expected))))
        worst.append(dict(index=list(map(int, index)), reference=expected, actual=observed,
                          reference_fp32_spacing=spacing,
                          delta_in_reference_spacings=maximum / spacing))
    return dict(max_abs=maximum, max_abs_hex=maximum.hex(),
                rms=float(np.sqrt(np.mean(delta ** 2))),
                max_reference_abs=float(np.max(np.abs(reference))),
                changed=int(np.count_nonzero(delta)), above_atol=int(np.sum(delta > atol)),
                atol=atol, passes=bool(np.all(delta <= atol)), worst=worst)


def cache_length(cache):
    if cache is None:
        return 0
    return int(cache.get_seq_length() if hasattr(cache, 'get_seq_length')
               else cache[0][0].shape[2])


class ForwardTrace:
    """Observe actual forward inputs, positions, masks, and pre-conversion dtypes."""
    def __init__(self, model, detailed=False):
        self.model, self.calls, self.values, self.handles = model, [], {}, []
        self.names = [f'transformer.h.{i}' for i in range(len(model.transformer.h))]
        if detailed:
            self.names = [name + suffix for name in self.names for suffix in
                          ('.ln_1', '.attn.c_attn', '.attn', '.ln_2', '.mlp', '')]
        self.names += ['transformer.ln_f']

    def __enter__(self):
        def before(module, args, kwargs):
            if args:
                raise ValueError('Diagnostic expects keyword model inputs')
            self.calls.append({
                key: value.detach().cpu().tolist() for key, value in kwargs.items()
                if torch.is_tensor(value)
            })
            self.calls[-1]['past_length'] = cache_length(kwargs.get('past_key_values'))
            self.calls[-1]['hook_dtypes'] = {}

        def positions(module, args):
            self.calls[-1]['effective_position_ids'] = args[0].detach().cpu().tolist()

        def attention(module, args, kwargs):
            mask = kwargs.get('attention_mask')
            summary = None
            if mask is not None:
                allowed = (mask if mask.dtype == torch.bool else mask == 0).cpu().numpy()
                positions = np.asarray(self.calls[-1]['effective_position_ids'][0])
                expected = np.arange(mask.shape[-1])[None, :] <= positions[:, None]
                summary = dict(dtype=str(mask.dtype), shape=list(mask.shape),
                    allowed_edges=int(allowed.sum()),
                    matches_expected_causal=bool(np.array_equal(allowed, expected[None, None, :, :])))
            self.calls[-1]['internal_attention_mask'] = summary

        self.handles.append(self.model.register_forward_pre_hook(before, with_kwargs=True))
        self.handles.append(self.model.transformer.wpe.register_forward_pre_hook(positions))
        self.handles.append(self.model.transformer.h[0].attn.register_forward_pre_hook(
            attention, with_kwargs=True))
        for name in self.names:
            def record(module, args, output, name=name):
                value = output[0] if isinstance(output, (tuple, list)) else output
                self.calls[-1]['hook_dtypes'][name] = str(value.dtype)
                # Preserve FP64 in controlled replays; .float() would hide evidence.
                last = value[:, -1, :].detach().cpu()
                if last.dtype == torch.bfloat16:
                    last = last.float()
                self.values.setdefault(name, []).append(last.numpy().copy()[0])
            self.handles.append(self.model.get_submodule(name).register_forward_hook(record))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.model = None


def causal_mask_matches(call):
    query_length = len(call['input_ids'][0])
    key_length = call['past_length'] + query_length
    mask = call['internal_attention_mask']
    if mask is None:
        # GPT2Attention uses is_causal iff q_len > 1 and mask is None.
        return query_length == key_length or query_length == 1
    return mask['matches_expected_causal'] and mask['shape'][-1] == key_length


def token_audit(legacy_calls, native_calls):
    """Check prefix stability, masks and positions, not merely input lengths."""
    if len(legacy_calls) != len(native_calls):
        raise ValueError('Forward call counts differ')
    prefix, rows = [], []
    for i, (legacy, native) in enumerate(zip(legacy_calls, native_calls)):
        if legacy['past_length'] == 0:
            prefix = []
        retained_length = len(prefix)
        prefix += legacy['input_ids'][0]
        full = native['input_ids'][0]
        new_length = len(legacy['input_ids'][0])
        expected_positions = list(range(len(full) - new_length, len(full)))
        rows.append(dict(row=i, native_length=len(full),
                         prefix_ids_equal=prefix == full,
                         cache_not_slid=legacy['past_length'] == retained_length,
                         legacy_mask_all_ones=all(legacy['attention_mask'][0]),
                         native_mask_all_ones=all(native['attention_mask'][0]),
                         legacy_mask_length=len(legacy['attention_mask'][0]),
                         legacy_causal_mask_correct=causal_mask_matches(legacy),
                         native_causal_mask_correct=causal_mask_matches(native),
                         native_positions_zero_based=native['effective_position_ids'][0] == list(range(len(full))),
                         legacy_positions_equal=legacy['effective_position_ids'][0] == expected_positions))
    return rows


def replay(model, calls, cached, *, detailed=True):
    """Replay recorded token IDs on one model, bypassing both wrappers."""
    past = None
    device = next(model.parameters()).device
    with ForwardTrace(model, detailed=detailed) as trace, torch.no_grad():
        for call in calls:
            tokens = {key: torch.tensor(call[key], dtype=torch.long, device=device)
                      for key in ('input_ids', 'attention_mask', 'token_type_ids', 'position_ids')
                      if key in call}
            output = model(**tokens, past_key_values=past if cached else None, use_cache=True)
            past = output.past_key_values if cached else None
    return trace


def compare_traces(reference, actual):
    return {name: difference(np.asarray(reference.values[name]), np.asarray(actual.values[name]))
            for name in reference.names}


def tiny_cpu():
    """Small executable counterexample to interpreting epsilon as an absolute bound."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from brainscore.model_helpers.text_wrapper import TextWrapper
    from brainscore_core.text import prepare_context
    from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

    torch.set_num_threads(1)
    vocab = {'[PAD]': 0, '[UNK]': 1, **{f'w{i}': i + 2 for i in range(97)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]',
                                       unk_token='[UNK]', model_max_length=128)
    parts = [' '.join(f'w{(i * 13 + j) % 97}' for j in range(n))
             for i, n in enumerate((13, 17, 11, 19))]
    prefixes = [prepare_context(parts[:i + 1]) for i in range(len(parts))]
    result = {}
    for implementation in ('eager', 'sdpa'):
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(17)
            config = GPT2Config(vocab_size=len(vocab), n_positions=128, n_embd=64,
                                n_layer=4, n_head=4, attn_implementation=implementation)
            model = GPT2LMHeadModel(config).to(dtype=dtype, device='cpu').eval()
            layer = 'transformer.h.3'
            with patch('torch.cuda.is_available', return_value=False), \
                    patch('torch.backends.mps.is_available', return_value=False):
                subject = HuggingfaceSubject('tiny', {'language_system': layer},
                                             model=model, tokenizer=tokenizer)
                subject.start_neural_recording('language_system', 'fMRI')
                with ForwardTrace(model, detailed=True) as legacy:
                    subject.digest_text(parts)
                wrapper = TextWrapper(model, tokenizer, batch_size=1, max_length=128)
                with ForwardTrace(model, detailed=True) as native:
                    wrapper(prefixes, [layer])
            full = replay(model, native.calls, cached=False)
            incremental = replay(model, legacy.calls, cached=True)
            result[f'{implementation}_{dtype}'] = dict(
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                token_audit=token_audit(legacy.calls, native.calls),
                cached_vs_full=compare_traces(legacy, native),
                native_vs_direct_full=compare_traces(native, full),
                legacy_vs_direct_cached=compare_traces(legacy, incremental),
                calls={'legacy': legacy.calls, 'native': native.calls})
    result['fp32_spacing'] = {str(x): float(np.spacing(np.float32(x)))
                              for x in (1, 512, 1024, 1536)}
    return result


def environment():
    import scipy
    import sklearn
    import transformers
    return dict(torch=torch.__version__, transformers=transformers.__version__,
                numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                cuda=torch.version.cuda, device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
                matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                default_dtype=str(torch.get_default_dtype()),
                autocast_cuda=torch.is_autocast_enabled('cuda'))


def observe_case(case, route, *, max_length=1024, score=False, score_float64=False):
    from brainscore import load_benchmark
    from brainscore.validation.benchmark_parity import local_candidate, _scalar
    from brainscore_core.metrics import Score
    from brainscore_language.utils.ceiling import ceiling_normalize

    candidate = local_candidate(case, route)
    model = candidate._preprocessors['text']._model if route == 'native' else (
        candidate._legacy.basemodel if route == 'adapter' else candidate.basemodel)
    tokenizer = candidate._preprocessors['text']._tokenizer if route == 'native' else (
        candidate._legacy.tokenizer if route == 'adapter' else candidate.tokenizer)
    if route == 'native':
        candidate._preprocessors['text']._max_length = max_length
    benchmark = load_benchmark(case.legacy if route == 'legacy' else case.unified)
    metric, captured, fits = benchmark.metric, [], []
    if score:
        original_fit = metric.regression.fit
        def record_fit(source, target):
            result = original_fit(source, target)
            estimator = metric.regression._regression
            fits.append(dict(dtype=str(source.dtype), shape=list(source.shape),
                             rank=int(estimator.rank_), singular_values=estimator.singular_.tolist()))
            return result
        metric.regression.fit = record_fit
    def capture(source, target):
        captured.append((source.transpose('presentation', 'neuroid').copy(deep=True), target))
        return metric(source, target) if score else Score(0.)
    benchmark.metric = capture
    try:
        with ForwardTrace(model) as trace:
            value = benchmark(candidate)
        if len(captured) != 1:
            raise ValueError('Diagnostic expects the fixed single-layer Pereira metric call')
        source, target = captured[0]
        metadata = dict(parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                        model_class=type(model).__name__, model_training=model.training,
                        checkpoint=str(getattr(model.config, '_name_or_path', '')),
                        is_quantized=bool(getattr(model, 'is_quantized', False)),
                        max_position_embeddings=model.config.max_position_embeddings,
                        attention_implementation=model.config._attn_implementation,
                        tokenizer_class=type(tokenizer).__name__,
                        tokenizer_max_length=tokenizer.model_max_length,
                        truncation_side=tokenizer.truncation_side, padding_side=tokenizer.padding_side,
                        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        score=None, raw=None, regression_fits=list(fits))
        if score:
            metadata.update(score=_scalar(value), raw=_scalar(value.attrs['raw']),
                            ceiling=_scalar(benchmark.ceiling))
        if score_float64:
            fits.clear()
            raw64 = metric(source.astype(np.float64), target)
            metadata['float64_regression'] = dict(raw=_scalar(raw64),
                score=_scalar(ceiling_normalize(raw64, benchmark.ceiling)), regression_fits=list(fits))
        # No truncation: audit lengths using the exact benchmark passage construction.
        from brainscore_core.text import prepare_context
        lengths = []
        passages = benchmark.data.passage_label.values
        for passage in sorted(set(passages)):
            parts = benchmark.data.stimulus.values[passages == passage].tolist()
            lengths += [len(tokenizer(prepare_context(parts[:i + 1]), truncation=False)['input_ids'])
                        for i in range(len(parts))]
        metadata['untruncated_lengths'] = lengths
        metadata['prefixes_over_512'] = sum(n > 512 for n in lengths)
        metadata['prefixes_over_1024'] = sum(n > 1024 for n in lengths)
        layer = f'transformer.h.{len(model.transformer.h) - 1}'
        metadata['hook_vs_metric_input'] = difference(np.asarray(trace.values[layer]), source.values, atol=0)
        return metadata, source.values.copy(), source.stimulus_id.values.astype(str), target.stimulus_id.values.astype(str), trace
    finally:
        benchmark.metric = metric
        if score:
            metric.regression.fit = original_fit
        if hasattr(candidate, 'reset'):
            candidate.reset()
        del candidate, benchmark, model
        gc.collect()


def ablate(case, legacy, native, worst_row):
    """Only replay the worst prefix's passage; no scoring or tolerance adjustment."""
    from brainscore.validation.benchmark_parity import local_candidate
    from torch.nn.attention import SDPBackend, sdpa_kernel
    start = worst_row
    while start and legacy.calls[start]['past_length']:
        start -= 1
    legacy_calls, native_calls = legacy.calls[start:worst_row + 1], native.calls[start:worst_row + 1]
    audit = token_audit(legacy_calls, native_calls)
    if not all(row['prefix_ids_equal'] and row['cache_not_slid'] for row in audit):
        return dict(skipped='Token mismatch or cache sliding requires semantic investigation first', audit=audit)
    result = {}
    layer = legacy.names[-2]  # Last block, before transformer.ln_f.
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_precision = torch.get_float32_matmul_precision()
    for mode in ('default', 'ieee_sdpa_math', 'ieee_eager', 'float64_eager'):
        subject = local_candidate(case, 'legacy')
        model = subject.basemodel
        context = nullcontext()
        try:
            if mode != 'default':
                torch.backends.cuda.matmul.allow_tf32 = False
                model.set_attn_implementation('sdpa' if mode == 'ieee_sdpa_math' else 'eager')
            if mode == 'ieee_sdpa_math':
                context = sdpa_kernel(SDPBackend.MATH)
            if mode == 'float64_eager':
                model.double()
            with context:
                cached = replay(model, legacy_calls, cached=True)
                full = replay(model, native_calls, cached=False)
            result[mode] = dict(layers=compare_traces(cached, full),
                                calls={'cached': cached.calls, 'full': full.calls})
            if mode == 'default':
                result[mode]['baseline_cached_reproduced'] = difference(
                    np.asarray(legacy.values[layer])[start:worst_row + 1],
                    np.asarray(cached.values[layer]), atol=0)
                result[mode]['baseline_full_reproduced'] = difference(
                    np.asarray(native.values[layer])[start:worst_row + 1],
                    np.asarray(full.values[layer]), atol=0)
        except Exception as error:
            result[mode] = {'error': repr(error)}
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original_tf32
            torch.set_float32_matmul_precision(original_precision)
            del subject, model
            gc.collect()
    return result


def gpu_run(args):
    from brainscore.validation.benchmark_parity import (
        CASES, PEREIRA_NATIVE_ULPS, PEREIRA_ULP_MAGNITUDE_FLOOR, PEREIRA_SCORE_ATOL,
        assert_pereira_activations, assert_pereira_score_drift,
    )
    if os.environ.get('RUN_UMI_PARITY') != '1' or os.environ.get('RESULTCACHING_DISABLE') != '1':
        raise RuntimeError('Requires RUN_UMI_PARITY=1 RESULTCACHING_DISABLE=1')
    if not Path(os.environ.get('UMI_PARITY_GPT2', '')).is_dir() or not os.environ.get('UMI_PARITY_GPT2'):
        raise RuntimeError('Set UMI_PARITY_GPT2 to the staged model/tokenizer directory')
    if not torch.cuda.is_available():
        raise RuntimeError('--gpu requires CUDA; use --tiny-cpu for offline synthetic diagnostics')
    report = dict(environment=environment(), cases={})
    for case in (case for case in CASES if case.domain == 'language'):
        runs, values, traces = {}, {}, {}
        case_report = dict(runs=runs, comparisons={})
        report['cases'][case.unified] = case_report
        for repeat in range(args.repeats):
            for label, route, length in [('legacy', 'legacy', 1024), ('adapter', 'adapter', 1024),
                                          ('native_1024', 'native', 1024), ('native_512', 'native', 512)]:
                key = f'{label}_{repeat}'
                meta, value, ids, target_ids, trace = observe_case(case, route, max_length=length,
                    score=args.score, score_float64=args.score_float64)
                if values:
                    np.testing.assert_array_equal(ids, reference_ids)
                    np.testing.assert_array_equal(target_ids, reference_target_ids)
                else:
                    reference_ids, reference_target_ids = ids, target_ids
                runs[key], values[key], traces[key] = meta, value, trace
                meta['calls'] = trace.calls
                np.savez_compressed(args.output / f'{case.unified}.{key}.npz',
                    activations=value, stimulus_id=ids,
                    **{name: np.asarray(array) for name, array in trace.values.items()})
                if key != 'legacy_0':
                    comp = dict(activations=difference(values['legacy_0'], value,
                                                       atol=0 if label == 'adapter' else 1e-6))
                    if label.startswith('native'):
                        activation = comp['activations']
                        activation['strict_absolute_check'] = {
                            key: activation.pop(key) for key in ('atol', 'above_atol', 'passes')}
                        activation['policy'] = dict(ulps=PEREIRA_NATIVE_ULPS,
                            magnitude_floor=PEREIRA_ULP_MAGNITUDE_FLOOR)
                        try:
                            activation['max_effective_fp32_ulps'] = assert_pereira_activations(
                                value, values['legacy_0'])
                            activation['passes'] = True
                        except AssertionError as error:
                            activation.update(passes=False, error=str(error))
                    if args.score:
                        comp['score_delta'] = meta['score'] - runs['legacy_0']['score']
                        comp['raw_delta'] = meta['raw'] - runs['legacy_0']['raw']
                        score_atol = (0 if label == 'adapter' else PEREIRA_SCORE_ATOL
                                      if label.startswith('native') else 1e-6)
                        raw_atol = score_atol * runs['legacy_0']['ceiling']
                        comp['score_atol'] = score_atol
                        comp['raw_atol'] = raw_atol
                        comp['score_passes'] = (abs(comp['score_delta']) <= score_atol
                                                and abs(comp['raw_delta']) <= raw_atol)
                    case_report['comparisons'][f'legacy_0_vs_{key}'] = comp
                # Flush after every scored route; a parity mismatch never discards scores.
                (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                print(case.unified, key, 'score=', meta['score'], 'raw=', meta['raw'], flush=True)
        case_report['max_length_512_vs_1024'] = difference(values['native_1024_0'], values['native_512_0'], atol=0)
        case_report['token_audit'] = token_audit(traces['legacy_0'].calls, traces['native_1024_0'].calls)
        case_report['repeatability'] = {label: [difference(values[f'{label}_0'], values[f'{label}_{i}'], atol=0)
            for i in range(1, args.repeats)] for label in ('legacy', 'adapter', 'native_1024', 'native_512')}
        if args.score:
            case_report['max_length_score_delta'] = {
                key: runs['native_512_0'][key] - runs['native_1024_0'][key] for key in ('score', 'raw')}
            case_report['score_repeat_ranges'] = {label: {
                key: float(np.ptp([runs[f'{label}_{i}'][key] for i in range(args.repeats)]))
                for key in ('score', 'raw')} for label in ('legacy', 'adapter', 'native_1024', 'native_512')}
        if args.ablations:
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            comparison = difference(values['legacy_0'], values['native_1024_0'])
            worst_row = comparison['worst'][0]['index'][0]
            case_report['ablations'] = ablate(case, traces['legacy_0'], traces['native_1024_0'], worst_row)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.score:
        report['pair_score_checks'] = []
        for repeat in range(args.repeats):
            pairs = [dict(benchmark=name, routes={
                'legacy': case['runs'][f'legacy_{repeat}'],
                'native': case['runs'][f'native_1024_{repeat}'],
            }) for name, case in report['cases'].items()]
            check = dict(repeat=repeat, passes=True)
            try:
                assert_pereira_score_drift(pairs)
            except AssertionError as error:
                check.update(passes=False, error=str(error))
            report['pair_score_checks'].append(check)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--tiny-cpu', action='store_true')
    mode.add_argument('--gpu', action='store_true')
    parser.add_argument('--score', action='store_true', help='Run real benchmark scoring (GPU mode only)')
    parser.add_argument('--score-float64', action='store_true', help='Also diagnose regression precision; does not change official scores')
    parser.add_argument('--ablations', action='store_true')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or (args.score_float64 and not args.score) or (args.tiny_cpu and (args.score or args.score_float64)):
        parser.error('Require repeats >= 1; --score-float64 needs --score; CPU mode never scores')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.tiny_cpu:
        report = tiny_cpu()
        (args.output / 'tiny-cpu.json').write_text(json.dumps(report, indent=2) + '\n')
    else:
        report = gpu_run(args)
        failed = any(not comp['activations']['passes'] or not comp.get('score_passes', True)
                     for case in report['cases'].values()
                     for name, comp in case['comparisons'].items() if 'native_512' not in name)
        failed = failed or any(not check['passes'] for check in report.get('pair_score_checks', []))
        print('Pereira regression criteria:', 'FAIL (diagnostics saved)' if failed else 'PASS')
        return int(failed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
