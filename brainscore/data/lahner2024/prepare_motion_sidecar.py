"""Build a motion-confound sidecar CSV for the TR-resolved Lahner2024 benchmark.

For each (subject, session, task, run) in ds005165 versionB, fetch the fmriprep
confounds.tsv and extract the 9 standard nuisance regressors:

    trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, framewise_displacement,
    csf, white_matter

Writes a long-format CSV with one row per (subject, session, task, run, tr_idx)
and columns above. Uploads to S3 alongside the existing assembly + events
sidecars. The runtime benchmark loads this sidecar and regresses the motion
columns out of BOLD per-run-per-voxel before z-scoring + ridge.

Usage:
    python -m brainscore.data.lahner2024.prepare_motion_sidecar \
        --output-path ~/lahner2024_prep/Lahner2024-fMRI-timeresolved-motion.csv \
        --upload-to-s3
"""
import argparse
import hashlib
import io
import sys
from pathlib import Path
from typing import List

import boto3
import pandas as pd

# Match the assembly-prep constants
SUBJECTS = [f'sub-{i:02d}' for i in range(1, 11)]
TASKS = ('test', 'train')
FMRIPREP_PREFIX = 'derivatives/versionB/fmriprep'
DS_BUCKET = 'openneuro.org'
DS_KEY_PREFIX = 'ds005165'

MOTION_COLS = [
    'trans_x', 'trans_y', 'trans_z',
    'rot_x',   'rot_y',   'rot_z',
    'framewise_displacement',
    'csf', 'white_matter',
]


def list_runs(subject: str, task: str, s3) -> List[dict]:
    """List all (session, run) pairs for a (subject, task) under fmriprep."""
    prefix = f'{DS_KEY_PREFIX}/{FMRIPREP_PREFIX}/{subject}/'
    runs = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=DS_BUCKET, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # match e.g. sub-01_ses-02_task-test_run-1_desc-confounds_timeseries.tsv
            if (f'task-{task}_' in key and 'desc-confounds_timeseries.tsv' in key
                    and not key.endswith('.json')):
                # parse session + run
                stem = Path(key).stem  # sub-01_ses-02_task-test_run-1_desc-confounds_timeseries
                parts = stem.split('_')
                ses = next(p for p in parts if p.startswith('ses-'))
                run = next(p for p in parts if p.startswith('run-'))
                runs.append({'session': ses, 'run': run, 'key': key})
    return runs


def fetch_confounds(key: str, s3, cache_dir: Path) -> pd.DataFrame:
    """Fetch one confounds.tsv from OpenNeuro S3 (cached locally)."""
    local = cache_dir / Path(key).name
    if not local.exists():
        s3.download_file(DS_BUCKET, key, str(local))
    df = pd.read_csv(local, sep='\t')
    # fmriprep marks first-row derivatives with 'n/a' — convert to numeric NaN.
    return df.apply(pd.to_numeric, errors='coerce')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path,
                        default=Path('/tmp/ds005165_motion_cache'))
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS)
    parser.add_argument('--tasks', nargs='+', default=list(TASKS))
    parser.add_argument('--upload-to-s3', action='store_true')
    parser.add_argument('--s3-bucket', type=str,
                        default='brainscore-storage/brainscore-vision/benchmarks/Lahner2024-fMRI')
    parser.add_argument('--s3-key', type=str,
                        default='Lahner2024-fMRI-timeresolved-motion.csv')
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client('s3', config=boto3.session.Config(signature_version=boto3.session.UNSIGNED)) \
        if False else boto3.client('s3')   # OpenNeuro is public; signed works fine
    # Use anon creds for OpenNeuro to avoid permission issues
    from botocore import UNSIGNED
    from botocore.config import Config
    s3_anon = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    rows = []
    for subject in args.subjects:
        for task in args.tasks:
            runs = list_runs(subject, task, s3_anon)
            print(f'  ({subject}, {task}): {len(runs)} runs', flush=True)
            for run_info in runs:
                df = fetch_confounds(run_info['key'], s3_anon, args.cache_dir)
                missing = [c for c in MOTION_COLS if c not in df.columns]
                if missing:
                    print(f'    WARN missing columns in {run_info["key"]}: {missing}',
                          flush=True)
                # Pad missing columns with 0 so downstream regression is well-defined.
                for c in missing:
                    df[c] = 0.0
                sub_df = df[MOTION_COLS].copy()
                sub_df['subject'] = subject
                sub_df['session'] = run_info['session']
                sub_df['task']    = task
                sub_df['run']     = run_info['run']
                sub_df['tr_idx']  = range(len(sub_df))
                rows.append(sub_df)

    print(f'\nConcatenating {len(rows)} runs of confounds...', flush=True)
    big = pd.concat(rows, ignore_index=True)
    # NaN handling: framewise_displacement first row is NaN by definition
    # (no previous frame to compare). Replace with 0 — equivalent to "no motion".
    big[MOTION_COLS] = big[MOTION_COLS].fillna(0.0)
    big.to_csv(args.output_path, index=False)
    print(f'Wrote {args.output_path} ({args.output_path.stat().st_size/1e6:.2f} MB, '
          f'{len(big)} rows)', flush=True)

    sha1 = hashlib.sha1(args.output_path.read_bytes()).hexdigest()
    print(f'sha1: {sha1}', flush=True)

    if args.upload_to_s3:
        parts = args.s3_bucket.split('/', 1)
        bucket_name, prefix = parts[0], (parts[1] if len(parts) > 1 else '')
        full_key = f'{prefix}/{args.s3_key}' if prefix else args.s3_key
        s3.upload_file(str(args.output_path), bucket_name, full_key)
        head = s3.head_object(Bucket=bucket_name, Key=full_key)
        v = head.get('VersionId', '')
        print('\n=== Data plugin values ===')
        print("Update unified/brainscore/data/lahner2024/__init__.py with:")
        print(f"timeresolved motion version: {v}")
        print(f"timeresolved motion sha1: {sha1}")


if __name__ == '__main__':
    main()
