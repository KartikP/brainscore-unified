"""``python -m brainscore.validation.run_parity`` -- the three-route check.

The parity suite is the standing guard against a whole class of defect: a model
route agreeing with the legacy reference on one code path while silently diverging
on another. Every route scores the legacy benchmark; the adapter and native routes
reach it through the legacy-benchmark facade, as ``brainscore.score`` does. That is not hypothetical. The Pereira
variants ran for months showing natively-registered models one bare sentence at
a time while adapter-routed models received the running passage context, and it
went unnoticed because the original validation only ever exercised the adapter.

It cannot run in ordinary CI: it needs the benchmark assemblies, real model
weights, and a few hours. So it is a release-time check, and this module exists
so that running it is one command rather than a remembered incantation.

    RESULTCACHING_DISABLE=1 python -m brainscore.validation.run_parity --resnet18 <path> --gpt2 <dir>

Staging the two reference checkpoints::

    python - <<'PY'
    import os, shutil, torch
    from huggingface_hub import snapshot_download
    from torchvision.models import resnet18, ResNet18_Weights
    os.makedirs('parity/gpt2')
    torch.save(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict(),
               'parity/resnet18.pt')
    names = ['config.json', 'merges.txt', 'model.safetensors', 'tokenizer.json',
             'tokenizer_config.json', 'vocab.json']
    snapshot = snapshot_download('gpt2', allow_patterns=names)
    for name in names:
        shutil.copy(os.path.join(snapshot, name), 'parity/gpt2')
    PY

The two FP32 policies accept only these six GPT-2 files as the Hub serves them
(``parity_runtime.CPU_GPT2_SHA256``). A directory written by ``save_pretrained`` has
other bytes and is refused. The benchmark data must be on the machine before the
run (load each benchmark once): the tool refuses a download.

Activation snapshots use temporary disk space and bounded comparison chunks.
The scoring metric still needs RAM. Peak memory and runtime of this revision
must be measured on the real cases; the earlier in-memory runner took hours.
Use run_vision_parity for sequential, resource-limited vision subprocesses.
"""

import argparse
import json
import os
import pathlib
import sys
import time


def main(argv=None):
    from brainscore.validation.benchmark_parity import (
        RELEASE_CASES, RIDGE_CASES, local_subject, validate_case,
        assert_pereira_score_drift, ParityFailure, POLICIES, HISTORICAL_POLICY, CPU_FP32_POLICY,
        L4_FP32_POLICY, FP32_POLICIES)
    from brainscore.validation.parity_runtime import checkpoint_manifest, execution, require_cpu_reference
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--resnet18', required=True,
                        help='torchvision ResNet18 state_dict file')
    parser.add_argument('--gpt2', required=True,
                        help='local Hugging Face GPT-2 model/tokenizer directory')
    parser.add_argument('--out', default='parity_report.json')
    parser.add_argument('--scratch-dir', default=None,
                        help='existing directory for temporary activation snapshots')
    parser.add_argument('--only', default='',
                        help='comma-separated benchmark identifiers; blank = all six release cases')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--language-precision', choices=('float32', 'float64-eager'), default='float32')
    parser.add_argument('--policy', choices=POLICIES, default=HISTORICAL_POLICY)
    args = parser.parse_args(argv)

    # The harness refuses to run against a warm cache, and it is right to: a
    # cached activation cannot testify about the route that produced it. The
    # caller turns the cache off; this command does not change it.
    if os.environ.get('RESULTCACHING_DISABLE') != '1':
        parser.error('Run with RESULTCACHING_DISABLE=1: a cached activation cannot testify '
                     'about the route that produced it')
    os.environ['RUN_UMI_PARITY'] = '1'
    os.environ['UMI_PARITY_RESNET18'] = str(pathlib.Path(args.resnet18).expanduser())
    os.environ['UMI_PARITY_GPT2'] = str(pathlib.Path(args.gpt2).expanduser())
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', BS_INSTALL_DEPENDENCIES='no')

    wanted = set(args.only.split(',')) if args.only else None
    known = {case.legacy for case in RELEASE_CASES}
    if wanted and wanted - known:
        parser.error(f'Unknown parity cases: {sorted(wanted - known)}')
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
        ridge = {case.legacy for case in RIDGE_CASES}
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
            result.setdefault('models', {}).setdefault(case.legacy, {})[route] = values
            if args.device != 'auto':
                actual = {device.split(':')[0] for device in values['parameter_devices']}
                if actual != {args.device}:
                    raise RuntimeError(f'Expected {args.device} model placement, observed {actual}')
            dtype = 'torch.float64' if case.domain == 'language' and args.language_precision == 'float64-eager' else 'torch.float32'
            if values['parameter_dtypes'] != [dtype]:
                raise RuntimeError(f'Expected {dtype} parameters, observed {values["parameter_dtypes"]}')
        return local_subject(case, route, language_precision=args.language_precision,
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
                if wanted and case.legacy not in wanted:
                    continue
                started = time.time()
                try:
                    reports[case.legacy] = validate_case(
                        case, factory, policy=args.policy, scratch_dir=args.scratch_dir,
                    )
                    print(f'  PASS {case.legacy} ({time.time() - started:.0f}s)', flush=True)
                except Exception as error:                  # noqa: BLE001
                    if isinstance(error, ParityFailure):
                        failed_observations[case.legacy] = error.report
                    failures[case.legacy] = f'{type(error).__name__}: {error}'
                    print(f'  FAIL {case.legacy} ({time.time() - started:.0f}s)\n'
                          f'       {failures[case.legacy]}', flush=True)
                save()
            names = {case.legacy for case in RIDGE_CASES}
            if args.policy in FP32_POLICIES and names <= reports.keys():
                try:
                    assert_pereira_score_drift([reports[name] for name in sorted(names)], policy=args.policy)
                except AssertionError as error:
                    failures['paired_drift:ridge'] = str(error)
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
