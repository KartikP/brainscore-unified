"""
Score 6 models on the M12-lite TR-resolved benchmark + report comparison vs
the existing GLM-beta variant.

Designed to run on EC2 with the bsu conda env. Each model is scored on BOTH
benchmarks for direct delta comparison.

Output: pandas DataFrame written to ~/lahner2024_scores.csv with columns
    model, benchmark, raw_r, mean_r, n_voxels, pipeline, time_sec

Usage:
    python -m experiments.lahner2024.score_timeresolved_all_models \
        [--models clip-vit-b-32 vjepa1-vitl] \
        [--variants timeresolved naturalistic]
"""
import argparse
import time
import traceback
from pathlib import Path

import pandas as pd

# Default model + benchmark sets
DEFAULT_MODELS = [
    'clip-vit-b-32',
    'qwen2.5-vl-3b',
    'blip2-opt-2.7b',
    'videomae-base',
    'vjepa1-vitl',
    'vjepa2-vitl',
]
DEFAULT_VARIANTS = ['naturalistic', 'naturalistic-visualROI',
                    'timeresolved', 'timeresolved-visualROI',
                    'timeresolved-improved', 'timeresolved-improved-visualROI']
VARIANT_BENCHMARK = {
    'naturalistic':                       'Lahner2024-fMRI-naturalistic',
    'naturalistic-visualROI':             'Lahner2024-fMRI-naturalistic-visualROI',
    'timeresolved':                       'Lahner2024-fMRI-naturalistic-timeresolved',
    'timeresolved-visualROI':             'Lahner2024-fMRI-naturalistic-timeresolved-visualROI',
    'timeresolved-improved':              'Lahner2024-fMRI-naturalistic-timeresolved-improved',
    'timeresolved-improved-visualROI':    'Lahner2024-fMRI-naturalistic-timeresolved-improved-visualROI',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--variants', nargs='+', default=DEFAULT_VARIANTS,
                        choices=list(VARIANT_BENCHMARK.keys()))
    parser.add_argument('--output-csv', type=Path,
                        default=Path('/home/ubuntu/lahner2024_scores.csv'))
    args = parser.parse_args()

    import brainscore

    rows = []
    for model_id in args.models:
        print(f"\n{'='*60}\n  Loading {model_id}\n{'='*60}", flush=True)
        try:
            t0 = time.time()
            model = brainscore.load_model(model_id)
            print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  FAIL load: {e}", flush=True)
            for v in args.variants:
                rows.append({'model': model_id, 'benchmark': VARIANT_BENCHMARK[v],
                             'raw_r': None, 'error': f'load: {e}'})
            continue

        for variant in args.variants:
            bench_id = VARIANT_BENCHMARK[variant]
            print(f"\n  scoring {bench_id}...", flush=True)
            try:
                bench = brainscore.load_benchmark(bench_id)
                t0 = time.time()
                score = bench(model)
                elapsed = time.time() - t0
                row = {
                    'model':     model_id,
                    'benchmark': bench_id,
                    'variant':   variant,
                    'raw_r':     float(score.attrs.get('raw', score)),
                    'mean_r':    float(score.attrs.get('mean_r', float('nan'))),
                    'n_voxels':  int(score.attrs.get('n_voxels_scored', 0)),
                    'pipeline':  str(score.attrs.get('pipeline', '?')),
                    'time_sec':  round(elapsed, 1),
                }
                rows.append(row)
                print(f"    raw r: {row['raw_r']:.4f} (n={row['n_voxels']}, "
                      f"pipeline={row['pipeline']}, {elapsed:.0f}s)", flush=True)
            except Exception as e:
                print(f"    FAIL: {e}", flush=True)
                traceback.print_exc()
                rows.append({'model': model_id, 'benchmark': bench_id,
                             'variant': variant, 'raw_r': None, 'error': str(e)})
            # Persist after every score so a partial run is still useful
            pd.DataFrame(rows).to_csv(args.output_csv, index=False)
            print(f"  (saved partial to {args.output_csv})", flush=True)

    # Final summary table
    print(f"\n\n{'='*70}\n  FINAL TABLE\n{'='*70}")
    df = pd.DataFrame(rows)
    if 'raw_r' in df.columns:
        pivot = df.pivot_table(index='model', columns='variant', values='raw_r')
        print(pivot.to_string())
    print(f"\nFull CSV: {args.output_csv}")


if __name__ == '__main__':
    main()
