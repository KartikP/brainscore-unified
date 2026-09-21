"""``python -m brainscore.validation.run_parity`` -- the three-route check.

The parity suite is the standing guard against a whole class of defect: a
`-unified` benchmark variant agreeing with its legacy counterpart on one code
path while silently diverging on another. That is not hypothetical. The Pereira
variants ran for months showing natively-registered models one bare sentence at
a time while adapter-routed models received the running passage context, and it
went unnoticed because the original validation only ever exercised the adapter.

It cannot run in ordinary CI: it needs the benchmark assemblies, real model
weights, and a few hours. So it is a release-time check, and this module exists
so that running it is one command rather than a remembered incantation.

    python -m brainscore.validation.run_parity --resnet18 <path> --gpt2 <dir>

Staging the two reference checkpoints, on a machine with a GPU::

    python - <<'PY'
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.save(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict(),
               'parity/resnet18.pt')
    AutoModelForCausalLM.from_pretrained('gpt2').save_pretrained('parity/gpt2')
    AutoTokenizer.from_pretrained('gpt2').save_pretrained('parity/gpt2')
    PY

Expect roughly three hours and about 10 GB of resident memory: every route's
activations are held for comparison, which is the point, and which is why a
15 GB box is not enough (it is OOM-killed part way through the vision cases).
"""

import argparse
import json
import os
import pathlib
import sys
import time


def main(argv=None):
    from brainscore.validation.benchmark_parity import (
        RELEASE_CASES, PEREIRA_CASES, RIDGE_CASES, local_candidate, validate_case,
        assert_pereira_score_drift, ParityFailure, POLICIES, HISTORICAL_POLICY, CPU_FP32_POLICY,
        L4_FP32_POLICY, FP32_POLICIES)
    from brainscore.validation.parity_runtime import checkpoint_manifest, execution, require_cpu_reference
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--resnet18', required=True,
                        help='torchvision ResNet18 state_dict file')
    parser.add_argument('--gpt2', required=True,
                        help='local Hugging Face GPT-2 model/tokenizer directory')
    parser.add_argument('--out', default='parity_report.json')
    parser.add_argument('--only', default='',
                        help='comma-separated benchmark identifiers; blank = all eight release cases')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--language-precision', choices=('float32', 'float64-eager'), default='float32')
    parser.add_argument('--policy', choices=POLICIES, default=HISTORICAL_POLICY)
    args = parser.parse_args(argv)

    # The harness refuses to run against a warm cache, and it is right to: a
    # cached activation cannot testify about the route that produced it.
    os.environ['RESULTCACHING_DISABLE'] = '1'
    os.environ['RUN_UMI_PARITY'] = '1'
    os.environ['UMI_PARITY_RESNET18'] = str(pathlib.Path(args.resnet18).expanduser())
    os.environ['UMI_PARITY_GPT2'] = str(pathlib.Path(args.gpt2).expanduser())
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', BS_INSTALL_DEPENDENCIES='no')

    wanted = set(args.only.split(',')) if args.only else None
    known = {case.unified for case in RELEASE_CASES}
    if wanted and wanted - known:
        parser.error(f'Unknown parity cases: {sorted(wanted - known)}')
    paired = {case.unified for case in PEREIRA_CASES}
    if wanted and wanted & paired and not paired <= wanted:
        parser.error('Select both Pereira linear cases to enforce the paired drift guard')
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.language_precision == 'float64-eager' and args.device != 'cpu':
        parser.error('The FP64 diagnostic reference requires --device cpu')
    if args.policy == CPU_FP32_POLICY:
        if args.device != 'cpu' or args.language_precision != 'float32':
            parser.error('The CPU FP32 policy requires --device cpu --language-precision float32')
    if args.policy == L4_FP32_POLICY:
        if args.device != 'cuda' or args.language_precision != 'float32':
            parser.error('The L4 FP32 policy requires --device cuda --language-precision float32')
    if args.policy in FP32_POLICIES:
        ridge = {case.unified for case in RIDGE_CASES}
        if wanted and wanted & ridge and not ridge <= wanted:
            parser.error('Select both Pereira ridge cases to enforce the FP32 paired drift guard')
    reports, failures, failed_observations = {}, {}, {}
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'schema_version': 2, 'policy': args.policy,
              'language_precision': args.language_precision,
              'reports': reports, 'failures': failures, 'failed_observations': failed_observations,
              'release_complete': False, 'selection_complete': False}
    def save():
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(output)
    def factory(case, route):
        def observe(values):
            result.setdefault('models', {}).setdefault(case.unified, {})[route] = values
            if args.device != 'auto':
                actual = {device.split(':')[0] for device in values['parameter_devices']}
                if actual != {args.device}:
                    raise RuntimeError(f'Expected {args.device} model placement, observed {actual}')
            dtype = 'torch.float64' if case.domain == 'language' and args.language_precision == 'float64-eager' else 'torch.float32'
            if values['parameter_dtypes'] != [dtype]:
                raise RuntimeError(f'Expected {dtype} parameters, observed {values["parameter_dtypes"]}')
        return local_candidate(case, route, language_precision=args.language_precision,
                               execution_observer=observe)
    try:
        result['checkpoints'] = checkpoint_manifest(os.environ['UMI_PARITY_RESNET18'], os.environ['UMI_PARITY_GPT2'])
        if args.policy in FP32_POLICIES:
            require_cpu_reference(result['checkpoints'])
        with execution(args.device, args.threads) as environment:
            result['environment'] = environment
            if args.policy == L4_FP32_POLICY and environment.get('cuda_name') != 'NVIDIA L4':
                raise ValueError(f'The L4 profile requires NVIDIA L4; observed {environment.get("cuda_name")}')
            save()
            for case in RELEASE_CASES:
                if wanted and case.unified not in wanted:
                    continue
                started = time.time()
                try:
                    reports[case.unified] = validate_case(case, factory, policy=args.policy)
                    print(f'  PASS {case.unified} ({time.time() - started:.0f}s)', flush=True)
                except Exception as error:                  # noqa: BLE001
                    if isinstance(error, ParityFailure):
                        failed_observations[case.unified] = error.report
                    failures[case.unified] = f'{type(error).__name__}: {error}'
                    print(f'  FAIL {case.unified} ({time.time() - started:.0f}s)\n'
                          f'       {failures[case.unified]}', flush=True)
                save()
            groups = (PEREIRA_CASES, RIDGE_CASES) if args.policy in FP32_POLICIES else (PEREIRA_CASES,)
            for group in groups:
                names = {case.unified for case in group}
                if names <= reports.keys():
                    try:
                        assert_pereira_score_drift([reports[name] for name in sorted(names)], policy=args.policy)
                    except AssertionError as error:
                        failures[f'paired_drift:{group[0].legacy.split("-")[-1]}'] = str(error)
    except Exception as error:                              # noqa: BLE001
        failures['preflight_or_runtime'] = f'{type(error).__name__}: {error}'
    result['selection_complete'] = not failures and set(reports) == (wanted or known)
    result['release_complete'] = (result['selection_complete'] and set(reports) == known
                                  and args.language_precision == 'float32')
    result['reference_only'] = args.language_precision != 'float32'
    save()
    print(f'\n{len(reports)} passed, {len(failures)} failed -> {args.out}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
