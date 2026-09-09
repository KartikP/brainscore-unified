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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--resnet18', required=True,
                        help='torchvision ResNet18 state_dict file')
    parser.add_argument('--gpt2', required=True,
                        help='local Hugging Face GPT-2 model/tokenizer directory')
    parser.add_argument('--out', default='parity_report.json')
    parser.add_argument('--only', default='',
                        help='comma-separated benchmark identifiers; blank = all six')
    args = parser.parse_args(argv)

    # The harness refuses to run against a warm cache, and it is right to: a
    # cached activation cannot testify about the route that produced it.
    os.environ['RESULTCACHING_DISABLE'] = '1'
    os.environ['RUN_UMI_PARITY'] = '1'
    os.environ['UMI_PARITY_RESNET18'] = str(pathlib.Path(args.resnet18).expanduser())
    os.environ['UMI_PARITY_GPT2'] = str(pathlib.Path(args.gpt2).expanduser())

    from brainscore.validation.benchmark_parity import CASES, local_candidate, validate_case

    wanted = set(args.only.split(',')) if args.only else None
    reports, failures = {}, {}
    for case in CASES:
        if wanted and case.unified not in wanted:
            continue
        started = time.time()
        try:
            reports[case.unified] = validate_case(case, local_candidate)
            print(f'  PASS {case.unified} ({time.time() - started:.0f}s)', flush=True)
        except Exception as error:                          # noqa: BLE001
            failures[case.unified] = f'{type(error).__name__}: {error}'
            print(f'  FAIL {case.unified} ({time.time() - started:.0f}s)\n'
                  f'       {failures[case.unified]}', flush=True)
        pathlib.Path(args.out).write_text(
            json.dumps({'reports': reports, 'failures': failures}, indent=1) + '\n')

    print(f'\n{len(reports)} passed, {len(failures)} failed -> {args.out}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
