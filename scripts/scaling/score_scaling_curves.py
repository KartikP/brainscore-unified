"""Score a model ladder (bad -> good) across registered capabilities and emit
scaling-curve-ready JSON, with null floors.

This is the organizing experiment: for each benchmark, run a ladder of models
from a random-init / chance null up to the strongest available model and record
the raw score. A benchmark worth running should climb across the ladder and sit
near its null for the random model. Run end-to-end and, when ``--no-cache`` is
set, clear the activation/result cache first so a past run can never inform the
current one (the user's explicit requirement).

Ladders are intentionally per-benchmark because compatibility differs (a text
model can't take a V4 image benchmark). Identifiers are the registered ones in
the unified package; pass overrides via CLI if they drift.

Run on EC2. Writes JSON to --out.
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np


# (benchmark_identifier, [model identifiers, worse -> better], null_model)
DEFAULT_LADDERS = {
    'MajajHong2015public.IT-pls-unified': {
        'models': ['random-vit-b-32', 'clip-vit-b-32', 'qwen2.5-vl-3b'],
        'null': 'random-vit-b-32',
        'capability': 'neural encoding (IT, r)',
    },
    'Pereira2018.243sentences-linear-unified': {
        'models': ['random-vit-b-32', 'clip-vit-b-32', 'gpt2', 'qwen2.5-vl-3b'],
        'null': 'random-vit-b-32',
        'capability': 'neural encoding (language, r)',
    },
    'Yeatman2021-lexical_decision-image': {
        'models': ['chance-baseline', 'random-vit-b-32', 'clip-vit-b-32', 'qwen2.5-vl-3b'],
        'null': 'chance-baseline',
        'capability': 'behavior (lexical decision)',
    },
}


def clear_caches():
    """Remove cached activations + scores so nothing is reused (true no-cache).

    Clears ~/.result_caching (the @store_xarray activation + score cache). Leaves
    ~/.brainio (downloaded stimulus sets + neural assemblies) intact — those are
    inputs, not results, and re-downloading them wouldn't make the test more
    'end-to-end', just slower.
    """
    rc = os.path.expanduser('~/.result_caching')
    if os.path.isdir(rc):
        shutil.rmtree(rc, ignore_errors=True)
        print(f"cleared {rc}", flush=True)
    else:
        print(f"no cache at {rc} (already clean)", flush=True)


def score_pair(model_id, benchmark_id):
    from brainscore import load_model, load_benchmark
    t0 = time.time()
    model = load_model(model_id)
    benchmark = load_benchmark(benchmark_id)
    score = benchmark(model)
    return _score_to_float(score), time.time() - t0


def _score_to_float(score):
    """Extract the center value from a Brain-Score Score robustly.

    A Score is an xarray DataAssembly; it may be a bare scalar, or carry an
    'aggregation' dim with a 'center' coordinate. Try the most specific form
    first, then fall back to the scalar value.
    """
    dims = getattr(score, 'dims', ())
    if 'aggregation' in dims:
        try:
            return float(score.sel(aggregation='center'))
        except Exception:
            pass
    try:
        return float(score)
    except (TypeError, ValueError):
        return float(np.asarray(score.values).ravel()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-cache', action='store_true',
                    help='clear result_caching before scoring (true end-to-end)')
    ap.add_argument('--benchmarks', nargs='*', default=None,
                    help='subset of benchmark identifiers to run')
    ap.add_argument('--out', default='/tmp/scaling_curves.json')
    args = ap.parse_args()

    if args.no_cache:
        clear_caches()

    ladders = DEFAULT_LADDERS
    if args.benchmarks:
        ladders = {k: v for k, v in ladders.items() if k in args.benchmarks}

    results = {'no_cache': args.no_cache, 'benchmarks': {}}
    for bench_id, spec in ladders.items():
        print(f"\n===== {bench_id} ({spec['capability']}) =====", flush=True)
        entry = {'capability': spec['capability'], 'null_model': spec['null'],
                 'scores': {}}
        for model_id in spec['models']:
            try:
                raw, dt = score_pair(model_id, bench_id)
                entry['scores'][model_id] = raw
                print(f"  {model_id:20s} -> {raw:.4f}  ({dt:.0f}s)", flush=True)
            except Exception as e:
                entry['scores'][model_id] = None
                print(f"  {model_id:20s} FAILED: {e}", flush=True)
                traceback.print_exc()
            # persist after every pair so a crash keeps partial results
            results['benchmarks'][bench_id] = entry
            with open(args.out, 'w') as f:
                json.dump(results, f, indent=2)

    print("\n=== SCALING SUMMARY ===")
    for bench_id, entry in results['benchmarks'].items():
        floor = entry['scores'].get(entry['null_model'])
        print(f"{bench_id} ({entry['capability']}) null={floor}")
        for m, s in entry['scores'].items():
            print(f"   {m:20s} {s}")
    print(f"written: {args.out}")


if __name__ == '__main__':
    sys.exit(main())
