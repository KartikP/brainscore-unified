"""
One-time data-prep script for the Lahner2024 TR-resolved fMRI assembly.

DESIGN: continuous-time encoding on fsaverage5 cortical surface.

Each (subject, run) produces one TR-resolved BOLD time-series. The assembly
treats each (subject, run) as a "presentation" — same xarray container as
the per-clip GLM-beta variant, but the presentation axis indexes runs (not
trials) and the time_bin axis spans the full run (not a per-trial window).

This avoids the rapid-event-related design's HRF-overlap problem (clips
every 4 s with TR=1.75 s leak into each other when per-trial windowed).
The continuous-time approach is what naturalistic-fMRI papers (Huth lab,
Friends/Sherlock benchmarks) use, and it's what TRIBEv2 was designed for.

## Output assembly shape

    (time_bin = N_TR_per_run_max,  neuroid = 20484,  presentation = 520)

where:
    - presentation = 520 = 10 subjects × 52 task-test/train runs per subject
    - neuroid = 20484 fsaverage5 cortical vertices (10242 × 2 hemispheres)
    - time_bin = max number of TRs across runs (shorter runs padded with NaN)

Per-presentation coords:
    subject:           sub-01 … sub-10
    session:           ses-02 … (BOLDMoments uses ses-02+ for video tasks)
    run:               run-1 … run-N
    task:              'test' or 'train'
    n_valid_TR:        actual TR count for this run (used to mask padding)

Per-time_bin coords:
    time_bin_start_ms:  TR start time in ms relative to run onset
    time_bin_end_ms:    TR end time

## Sidecar events file

Per-run stimulus events live in a separate CSV alongside the .nc:

    Lahner2024-fMRI-timeresolved-events.csv
        columns: subject, session, run, trial_idx, stimulus_id, onset_sec, duration_sec, trial_type

The benchmark loads BOTH artifacts at runtime. Ridge regression uses the
events to build a per-run feature time-series at TR resolution from the
model's per-stimulus features.

## Confirmed parameters (from EC2 reconnaissance, ds005165 v1.0.4)

- TR = 1.75 s
- Tasks: 'train' (1000 stimuli × 3 reps) + 'test' (102 stimuli × 10 reps)
- Stimulus design: 3 s clip + 1 s ISI = 4 s SOA, 113 trials per run
- Some 'oddball' trials (stim_file='n/a', trial_type='oddball') — exclude from analysis
- 52 total task-test/train runs per subject across multiple sessions
- 10 subjects (sub-01 … sub-10)
- fmriprep outputs in derivatives/versionB/fmriprep/<sub>/<ses>/func/
    - format: hemi-L/R_space-fsaverage_bold.func.gii (~190 MB per hemi per run)
    - resolution: full fsaverage (~163k vertices/hemi) — must downsample to fsaverage5

## Total download estimate

52 runs × 10 subjects × 2 hemis × ~190 MB = ~200 GB raw download.
After per-run downsample to fsaverage5 + per-run aggregation, intermediate
on-disk footprint stays under ~5 GB. The final .nc artifact is ~5-10 GB.

## Dependencies (install on EC2 before running)

    pip install nilearn nibabel pandas numpy xarray boto3 neuromaps
    # nilearn for surface-to-surface resampling, nibabel for gifti loading

## Usage

    # On EC2, in an env with the deps above:
    python prepare_timeresolved_assembly.py \\
        --output-path /tmp/lahner2024_timeresolved.nc \\
        --events-path /tmp/lahner2024_timeresolved-events.csv \\
        --upload-to-s3
"""

import argparse
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# ── Confirmed constants (from EC2 recon) ──────────────────────────────

OPENNEURO_BASE = 's3://openneuro.org/ds005165'

# Full BOLDMoments cohort
SUBJECTS = [f'sub-{i:02d}' for i in range(1, 11)]   # sub-01 … sub-10

# Tasks we actually care about
TASKS = ('test', 'train')                # NOT localizer or rest

# Confirmed scanner / paradigm parameters
TR_SEC = 1.75
SOA_SEC = 4.0                            # stimulus onset asynchrony
CLIP_DURATION_SEC = 3.0
N_TRIALS_PER_RUN = 113                   # before oddball exclusion
N_RUNS_PER_SUBJECT_TOTAL = 52            # train + test combined

# Surface space (matches existing GLM-beta assembly)
N_VERTICES_PER_HEMI = 10242              # fsaverage5 standard
N_VERTICES_TOTAL    = 2 * N_VERTICES_PER_HEMI    # 20484

# Padding policy: assume runs vary by ~5 TRs around the median.
# We compute max actual TR count after first subject's data is read,
# then pad shorter runs with NaN. n_valid_TR coord tracks per-run length.

# Source-space derivatives path on OpenNeuro S3
FMRIPREP_PREFIX = 'derivatives/versionB/fmriprep'


# ── Top-level orchestration ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-path', type=Path, required=True,
                        help='Where to write the .nc neural assembly.')
    parser.add_argument('--events-path', type=Path, required=True,
                        help='Where to write the .csv events sidecar.')
    parser.add_argument('--data-cache', type=Path,
                        default=Path('/tmp/ds005165_cache'),
                        help='Where to cache downloaded OpenNeuro files.')
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS)
    parser.add_argument('--tasks', nargs='+', default=list(TASKS),
                        help="Which task names to include (default: test + train).")
    parser.add_argument('--upload-to-s3', action='store_true')
    parser.add_argument('--s3-bucket', type=str,
                        default='brainscore-storage/brainscore-vision/benchmarks/Lahner2024-fMRI')
    parser.add_argument('--s3-key-assembly', type=str,
                        default='Lahner2024-fMRI-timeresolved.nc')
    parser.add_argument('--s3-key-events', type=str,
                        default='Lahner2024-fMRI-timeresolved-events.csv')
    args = parser.parse_args()

    # 1. For each subject: extract all runs for all requested tasks, then DELETE
    #    the per-subject gii cache before moving to the next subject. Per-subject
    #    cache footprint is ~20GB (52 runs × ~380MB L+R each); without cleanup
    #    the full 10-subject run would need ~200GB free.
    import shutil
    run_records = []
    for subject in args.subjects:
        subj_cache = args.data_cache / 'derivatives' / 'versionB' / 'fmriprep' / subject
        for task in args.tasks:
            run_records.extend(extract_subject_task_runs(
                subject=subject, task=task, data_cache=args.data_cache))
        # Purge giis for this subject — extracted time-series already in records[].
        # Keep events.tsv files (small) under the raw cache path.
        if subj_cache.exists():
            print(f"  cleanup: removing {subj_cache} (~20GB)", flush=True)
            shutil.rmtree(subj_cache, ignore_errors=True)

    if not run_records:
        raise SystemExit(f"No runs extracted for subjects={args.subjects} tasks={args.tasks}.")

    # 2. Build the assembly + events DataFrame.
    assembly = build_assembly(run_records)
    events_df = build_events_table(run_records)

    # 3. Save locally.
    assembly.to_netcdf(args.output_path)
    events_df.to_csv(args.events_path, index=False)
    sha1_assembly = compute_sha1(args.output_path)
    sha1_events = compute_sha1(args.events_path)
    print(f"\nWrote assembly:  {args.output_path}  ({args.output_path.stat().st_size/1e9:.2f} GB)")
    print(f"  sha1: {sha1_assembly}")
    print(f"Wrote events:    {args.events_path}  ({args.events_path.stat().st_size/1e6:.1f} MB)")
    print(f"  sha1: {sha1_events}")

    # 4. Upload.
    if args.upload_to_s3:
        v_assembly = upload_to_s3(args.output_path, args.s3_bucket, args.s3_key_assembly)
        v_events   = upload_to_s3(args.events_path, args.s3_bucket, args.s3_key_events)
        print("\n=== Paste into benchmark_timeresolved.py ===")
        print(f"TIMERESOLVED_ASSEMBLY_VERSION_ID = '{v_assembly}'")
        print(f"TIMERESOLVED_ASSEMBLY_SHA1       = '{sha1_assembly}'")
        print(f"TIMERESOLVED_EVENTS_VERSION_ID   = '{v_events}'")
        print(f"TIMERESOLVED_EVENTS_SHA1         = '{sha1_events}'")


# ── Per-(subject, task) extraction ────────────────────────────────────

def extract_subject_task_runs(
    subject: str,
    task: str,
    data_cache: Path,
) -> List[dict]:
    """Per (subject, task), iterate runs across sessions; for each run download
    fsaverage L+R surface BOLD + events.tsv, downsample to fsaverage5, package
    a single record.

    Returns a list of dicts, one per run, with:
        'subject':      'sub-01'
        'session':      'ses-02'
        'run':          'run-1'
        'task':         'test' / 'train'
        'time_series':  np.ndarray (n_TR, 20484) on fsaverage5
        'events':       pd.DataFrame from events.tsv (filtered to non-oddball trials)
        'n_TR':         int — len(time_series)
    """
    import subprocess

    data_cache.mkdir(parents=True, exist_ok=True)

    # 1. List sessions for this subject under fmriprep derivatives
    fmriprep_root = f'{OPENNEURO_BASE}/{FMRIPREP_PREFIX}/{subject}'
    cp = subprocess.run(
        ['aws', 's3', 'ls', '--no-sign-request', '--recursive', f'{fmriprep_root}/'],
        capture_output=True, text=True, check=True,
    )
    # Filter to L-hemi gii files for this task (each run has paired L+R)
    pattern_L = f'_task-{task}_run-'
    pattern_L_suffix = '_hemi-L_space-fsaverage_bold.func.gii'
    run_keys_L = []
    for line in cp.stdout.splitlines():
        path = line.strip().split()[-1]
        if pattern_L in path and pattern_L_suffix in path:
            run_keys_L.append(path)
    print(f"  ({subject}, {task}): found {len(run_keys_L)} runs", flush=True)

    records = []
    for L_key in sorted(run_keys_L):
        # Parse out (session, run) from the key path
        # ds005165/derivatives/versionB/fmriprep/sub-01/ses-02/func/sub-01_ses-02_task-test_run-1_hemi-L_space-fsaverage_bold.func.gii
        basename = Path(L_key).name
        parts = basename.split('_')
        session = next(p for p in parts if p.startswith('ses-'))
        run = next(p for p in parts if p.startswith('run-'))
        R_key = L_key.replace('_hemi-L_', '_hemi-R_')

        # 2. Find the matching raw events.tsv — under raw BIDS (NOT derivatives)
        events_key = (f'ds005165/{subject}/{session}/func/'
                      f'{subject}_{session}_task-{task}_{run}_events.tsv')

        # 3. Download all three (cached locally).
        L_local = _s3_download(L_key, data_cache)
        R_local = _s3_download(R_key, data_cache)
        events_local = _s3_download(events_key, data_cache)

        # 4. Load gii → (n_TR, n_vertices_full_fsaverage), downsample to fsaverage5
        ts_L = _load_gii_and_downsample(L_local)   # (n_TR, 10242)
        ts_R = _load_gii_and_downsample(R_local)
        assert ts_L.shape == ts_R.shape, f"hemi shape mismatch: {ts_L.shape} vs {ts_R.shape}"
        ts_LR = np.concatenate([ts_L, ts_R], axis=1).astype(np.float32)  # (n_TR, 20484)

        # 5. Parse events.tsv, filter, derive stimulus_id from stim_file
        events_df = pd.read_csv(events_local, sep='\t')
        events_df = events_df[events_df['trial_type'].isin(['test', 'train'])].copy()
        # stim_file looks like 'test/1074.mp4' or 'train/0123.mp4' → 'test_1074'
        events_df['stimulus_id'] = events_df['stim_file'].apply(
            lambda s: f"{Path(s).parent.name}_{Path(s).stem}" if isinstance(s, str) else None)
        events_df = events_df[events_df['stimulus_id'].notna()].reset_index(drop=True)

        records.append({
            'subject':     subject,
            'session':     session,
            'run':         run,
            'task':        task,
            'time_series': ts_LR,
            'events':      events_df,
            'n_TR':        ts_LR.shape[0],
        })
        print(f"    {subject}/{session}/{task}/{run}: "
              f"{ts_LR.shape[0]} TRs, {len(events_df)} non-oddball trials", flush=True)

    return records


def _s3_download(s3_key: str, cache_dir: Path) -> Path:
    """Cached download from openneuro.org S3. Returns local path."""
    import subprocess
    # s3_key may be path-like (no scheme) — strip leading "ds005165/" if present
    if s3_key.startswith('ds005165/'):
        relative = s3_key[len('ds005165/'):]
    else:
        relative = s3_key
    local_path = cache_dir / relative
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['aws', 's3', 'cp', '--no-sign-request',
         f's3://openneuro.org/ds005165/{relative}', str(local_path), '--quiet'],
        check=True,
    )
    return local_path


def _load_gii_and_downsample(gii_path: Path) -> np.ndarray:
    """Load a fsaverage gifti time-series and downsample to fsaverage5.

    Convention: fsaverage5 vertices are a hierarchical icosahedral subdivision
    of fsaverage7 (full fsaverage). Specifically, fsaverage5's 10242 vertices
    correspond to the first 10242 vertices of fsaverage(7)'s 163842 — the
    earlier vertices in the sequence are the lower-resolution mesh.

    This is a property of FreeSurfer's recursive icosahedral mesh construction.
    Verified against neuromaps internally before adopting.

    Returns: (n_TR, 10242) — n_TR same as the source gii.
    """
    import nibabel as nib
    img = nib.load(str(gii_path))
    # gifti agg_data() for time-series surfaces returns a tuple of arrays per
    # darray — one per TR, each of shape (n_vertices_full,). Stack to
    # (n_TR, n_vertices_full).
    data_per_tr = img.agg_data()
    if isinstance(data_per_tr, tuple):
        # multiple darrays — one per TR
        ts_full = np.stack(data_per_tr, axis=0)
    else:
        # single 2-D darray
        ts_full = np.asarray(data_per_tr)
        if ts_full.shape[0] > ts_full.shape[1]:
            # Heuristic: more rows than cols → likely (n_vertices, n_TR); transpose.
            ts_full = ts_full.T
    # Downsample to fsaverage5 — first 10242 vertices.
    return ts_full[:, :N_VERTICES_PER_HEMI]


# ── Building the assembly ─────────────────────────────────────────────

def build_assembly(run_records: List[dict]) -> xr.DataArray:
    """Pad runs to common length, stack into (time_bin, neuroid, presentation).

    Padding policy: time_bin axis = max n_TR across all runs. Shorter runs
    are padded with NaN. n_valid_TR coord on presentation tracks per-run
    actual length so the benchmark can mask padding when computing metrics.
    """
    from brainscore_core.supported_data_standards.brainio.assemblies import (
        NeuronRecordingAssembly,
    )

    n_TR_max = max(r['n_TR'] for r in run_records)
    n_runs = len(run_records)
    n_neuroid = N_VERTICES_TOTAL

    # Pre-allocate padded array; fill from records.
    arr = np.full((n_TR_max, n_neuroid, n_runs), np.nan, dtype=np.float32)
    for i, r in enumerate(run_records):
        arr[:r['n_TR'], :, i] = r['time_series']

    # Coords
    time_bin_start_ms = (np.arange(n_TR_max) * TR_SEC * 1000.0).astype(float)
    time_bin_end_ms   = time_bin_start_ms + TR_SEC * 1000.0

    neuroid_id   = ([f'fsaverage5.lh.{i}' for i in range(N_VERTICES_PER_HEMI)] +
                    [f'fsaverage5.rh.{i}' for i in range(N_VERTICES_PER_HEMI)])
    hemisphere   = ['L'] * N_VERTICES_PER_HEMI + ['R'] * N_VERTICES_PER_HEMI

    presentation_id = [f"{r['subject']}_{r['session']}_{r['task']}_{r['run']}"
                       for r in run_records]

    return NeuronRecordingAssembly(
        arr,
        coords={
            'time_bin_start_ms': ('time_bin', time_bin_start_ms),
            'time_bin_end_ms':   ('time_bin', time_bin_end_ms),
            'neuroid_id':        ('neuroid', neuroid_id),
            'hemisphere':        ('neuroid', hemisphere),
            'presentation_id':   ('presentation', presentation_id),
            'subject':           ('presentation', [r['subject'] for r in run_records]),
            'session':           ('presentation', [r['session'] for r in run_records]),
            'task':              ('presentation', [r['task']    for r in run_records]),
            'run':               ('presentation', [r['run']     for r in run_records]),
            'n_valid_TR':        ('presentation', [r['n_TR']    for r in run_records]),
        },
        dims=['time_bin', 'neuroid', 'presentation'],
    )


def build_events_table(run_records: List[dict]) -> pd.DataFrame:
    """Long-format CSV: one row per (subject, session, run, trial)."""
    rows = []
    for r in run_records:
        events = r['events']
        for trial_idx, ev in events.iterrows():
            rows.append({
                'subject':      r['subject'],
                'session':      r['session'],
                'run':          r['run'],
                'task':         r['task'],
                'trial_idx':    trial_idx,
                'stimulus_id':  ev['stimulus_id'],
                'onset_sec':    float(ev['onset']),
                'duration_sec': float(ev['duration']),
                'trial_type':   ev['trial_type'],
            })
    return pd.DataFrame(rows)


# ── S3 + hashing ──────────────────────────────────────────────────────

def compute_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def upload_to_s3(local_path: Path, bucket: str, key: str) -> str:
    """Upload, return the new version_id."""
    import boto3
    parts = bucket.split('/', 1)
    bucket_name = parts[0]
    key_prefix = parts[1] if len(parts) > 1 else ''
    full_key = f'{key_prefix}/{key}' if key_prefix else key

    s3 = boto3.client('s3')
    with open(local_path, 'rb') as f:
        response = s3.put_object(Bucket=bucket_name, Key=full_key, Body=f)
    return response.get('VersionId', '')


if __name__ == '__main__':
    main()
