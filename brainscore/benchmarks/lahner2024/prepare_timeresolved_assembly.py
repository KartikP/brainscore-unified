"""
One-time data-prep script for the Lahner2024 TR-resolved fMRI assembly.

Runs on EC2 (needs ~50GB temporary disk for the OpenNeuro download). Reads
fMRIPrep-preprocessed time-series + events.tsv from OpenNeuro ds005165,
extracts a TR-resolved BOLD response per (clip, repetition, TR), and uploads
the resulting NeuronRecordingAssembly to brain-score's S3 bucket so the
runtime benchmark loads it via the standard `load_assembly_from_s3()` path.

Output assembly shape (target):
    (time_bin=N_TR_per_clip, neuroid=20484, presentation=10260)

where:
    - presentation = 1026 stimuli × 10 reps  (matches existing GLM-beta variant)
    - neuroid = 20484 fsaverage5 cortical vertices (10242 × 2 hemispheres)
    - time_bin = N_TR_per_clip  (depends on TR + per-clip window — typically 4-8 TRs
        spanning clip onset → clip end + post-stimulus HRF window)

This script is **not** imported by the runtime benchmark. It produces an
artifact (a .nc file uploaded to S3); the runtime benchmark just downloads
that artifact via load_assembly_from_s3().

Steps (high level):
    1. For each subject (sub-01 … sub-10):
         a. Download fMRIPrep-preprocessed BOLD on fsaverage5 (left + right hemis)
         b. Download events.tsv per run for stimulus onset times
         c. For each trial: extract a TR-aligned window starting at onset
         d. Stack windows into (n_trials_subject, N_TR_per_clip, 20484)
    2. Concatenate across subjects → (10260, N_TR_per_clip, 20484)
    3. Reorder to (N_TR_per_clip, 20484, 10260) and wrap as NeuronRecordingAssembly
        with stimulus_id, repetition, subject coords on the presentation axis,
        and time_bin_start_ms / time_bin_end_ms coords on the time_bin axis.
    4. Save as .nc, compute sha1, upload to S3 with a versioned key.
    5. Print the (version_id, sha1) tuple for pasting into benchmark.py.

## Unknowns to resolve at runtime (when this script first runs)

- **TR**: read from `sub-01_ses-01_task-bmd_run-01_bold.json` once data is downloaded.
  Almost certainly 1.5s for Siemens Prisma multiband; confirm and bake in.
- **Per-clip window**: clip is 3s. We need to extend the window to capture the
  HRF-delayed response — peak at ~5-6s post-stimulus, settling by ~12-15s.
  At TR=1.5s this is ~10 TRs. At TR=1s this is ~15 TRs. Pick a fixed window
  that covers HRF and document the choice.
- **Exact BIDS task name**: probably `task-bmd` or `task-test`/`task-train` per the
  README's mention of "version A vs version B" data products. Confirm by listing
  the dataset structure.
- **Trial onset alignment**: events.tsv gives onset in seconds since run start.
  Need to convert to TR index via floor(onset / TR). If multiple trials fall in
  the same TR (rare with proper jittered design), we need a policy — probably
  exclude or collapse.
- **Surface space**: BOLDMoments versionB releases data in fsaverage and fsaverage5.
  Use fsaverage5 to match the 20484-vertex existing assembly. If only fsaverage7
  is available we'd need to downsample.

## Dependencies (install on EC2 before running)

    pip install nilearn nibabel pandas numpy xarray boto3
    # Plus brainio (already installed) and the brain-score-unified env

## Usage

    # On EC2, in an env with the deps above:
    python prepare_timeresolved_assembly.py --output-path /tmp/lahner2024_timeresolved.nc \
                                            --upload-to-s3 \
                                            --s3-bucket brainscore-storage/brainscore-vision/benchmarks/Lahner2024-fMRI \
                                            --s3-key Lahner2024-fMRI-timeresolved.nc
"""

import argparse
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# ── Constants ─────────────────────────────────────────────────────────

OPENNEURO_BASE = 'https://s3.amazonaws.com/openneuro.org/ds005165'

# Full BOLDMoments cohort
SUBJECTS = [f'sub-{i:02d}' for i in range(1, 11)]   # sub-01 … sub-10

# Number of stimuli
N_TRAIN_VIDEOS = 1000
N_TEST_VIDEOS  = 102
N_VIDEOS_TOTAL = N_TRAIN_VIDEOS + N_TEST_VIDEOS    # 1102

# Repetitions
N_TRAIN_REPS = 3
N_TEST_REPS  = 10

# Existing GLM-beta variant uses 1026 stimuli (subset with annotations).
# We use the same subset so presentation axes align across variants.
EXPECTED_N_STIMULI_USED = 1026
EXPECTED_N_PRESENTATIONS = EXPECTED_N_STIMULI_USED * 10   # 10260

# Surface space (matches existing assembly)
N_VERTICES_PER_HEMI = 10242
N_VERTICES_TOTAL    = 2 * N_VERTICES_PER_HEMI    # 20484

# Stimulus / TR window
CLIP_DURATION_SEC = 3.0
HRF_TAIL_SEC = 9.0   # extra window after clip offset to capture HRF peak + decay
PER_CLIP_WINDOW_SEC = CLIP_DURATION_SEC + HRF_TAIL_SEC   # 12s — covers HRF


# ── Top-level orchestration ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-path', type=Path, required=True,
                        help='Where to write the .nc assembly file locally.')
    parser.add_argument('--data-cache', type=Path,
                        default=Path('/tmp/ds005165_cache'),
                        help='Where to cache downloaded OpenNeuro files.')
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS,
                        help='Subject IDs to include (default: all 10).')
    parser.add_argument('--upload-to-s3', action='store_true',
                        help='Upload the resulting .nc to S3.')
    parser.add_argument('--s3-bucket', type=str,
                        default='brainscore-storage/brainscore-vision/benchmarks/Lahner2024-fMRI',
                        help='Target S3 bucket path.')
    parser.add_argument('--s3-key', type=str,
                        default='Lahner2024-fMRI-timeresolved.nc',
                        help='Target S3 key.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Skip actual download / upload; print plan only.')
    args = parser.parse_args()

    # 1. Discover TR + BIDS structure from one subject's bold.json sidecar.
    tr_sec, task_name = discover_tr_and_task(args.subjects[0], args.data_cache,
                                             dry_run=args.dry_run)
    print(f"Discovered TR = {tr_sec}s, task = {task_name!r}")

    n_tr_per_window = int(np.ceil(PER_CLIP_WINDOW_SEC / tr_sec))
    print(f"Per-clip window = {PER_CLIP_WINDOW_SEC}s → {n_tr_per_window} TRs")

    # 2. Per-subject: download + extract per-trial windows.
    subject_data = []   # list of (n_trials_i, n_tr, n_vertices) arrays + metadata
    for subject in args.subjects:
        print(f"\n=== {subject} ===")
        windows, trial_meta = extract_subject_trial_windows(
            subject=subject,
            data_cache=args.data_cache,
            tr_sec=tr_sec,
            n_tr_per_window=n_tr_per_window,
            task_name=task_name,
            dry_run=args.dry_run,
        )
        subject_data.append((subject, windows, trial_meta))

    # 3. Concatenate across subjects, build the xarray NeuronRecordingAssembly.
    assembly = build_assembly(subject_data, tr_sec=tr_sec,
                              n_tr_per_window=n_tr_per_window)

    # 4. Save locally.
    assembly.to_netcdf(args.output_path)
    sha1 = compute_sha1(args.output_path)
    print(f"\nWrote {args.output_path} ({args.output_path.stat().st_size/1e6:.1f} MB)")
    print(f"sha1: {sha1}")

    # 5. Upload to S3.
    if args.upload_to_s3:
        version_id = upload_to_s3(args.output_path, args.s3_bucket, args.s3_key)
        print(f"\nUploaded to s3://{args.s3_bucket}/{args.s3_key}")
        print(f"version_id: {version_id}")
        print(f"sha1: {sha1}")
        print("\nPaste these into benchmark.py:")
        print(f"    TIMERESOLVED_ASSEMBLY_VERSION_ID = '{version_id}'")
        print(f"    TIMERESOLVED_ASSEMBLY_SHA1 = '{sha1}'")


# ── Stage 1: discover TR + task name ──────────────────────────────────

def discover_tr_and_task(subject: str, data_cache: Path,
                         dry_run: bool = False) -> Tuple[float, str]:
    """Download one bold.json sidecar and read RepetitionTime + task name.

    The BIDS structure is dataset_root/<subject>/<session>/func/<subject>_<session>_task-<X>_run-<Y>_bold.json.
    We probe for the first task by listing the OpenNeuro file index — or, if listing
    isn't easily available, try a small set of known task name candidates.
    """
    raise NotImplementedError(
        "TODO on EC2: list ds005165/<subject>/ses-*/func/ via S3, "
        "find the first *_bold.json, fetch it, parse RepetitionTime "
        "and the task-<NAME> portion of the filename. Return (tr, task_name)."
    )


# ── Stage 2: per-subject extraction ───────────────────────────────────

def extract_subject_trial_windows(
    subject: str,
    data_cache: Path,
    tr_sec: float,
    n_tr_per_window: int,
    task_name: str,
    dry_run: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Per-subject: download fMRIPrep BOLD + events, extract TR-aligned trial windows.

    For each run:
        1. Download <subject>_<session>_task-<task>_run-<R>_space-fsaverage5_hemi-L_bold.func.gii
        2. Download corresponding events.tsv
        3. For each row in events.tsv with trial_type='clip' (or whatever Lahner used):
             a. Compute starting TR: floor(onset_sec / tr_sec)
             b. Slice BOLD[starting_tr : starting_tr + n_tr_per_window, :]
             c. Pad with NaN if at end of run
        4. Stack windows + record (stimulus_id, repetition, run_id, subject) per window

    Returns:
        windows: (n_trials_subject, n_tr_per_window, 20484)
        trial_meta: DataFrame with columns [stimulus_id, repetition, run_id, subject]
    """
    raise NotImplementedError(
        "TODO on EC2:\n"
        "  - List runs for this subject\n"
        "  - For each run: download L+R hemi fMRIPrep gifti files\n"
        "  - Concatenate hemis along vertex axis (L first, then R) → (n_tr_run, 20484)\n"
        "  - Download events.tsv\n"
        "  - For each clip trial: TR-aligned slice into windows array\n"
        "  - Track (stimulus_id, repetition_idx, run_id, subject) metadata\n"
        "  - Filter to stimuli that appear in our 1026-video subset (matching existing variant)"
    )


# ── Stage 3: build the assembly ───────────────────────────────────────

def build_assembly(
    subject_data: List[Tuple[str, np.ndarray, pd.DataFrame]],
    tr_sec: float,
    n_tr_per_window: int,
) -> xr.DataArray:
    """Concatenate per-subject data into a single NeuronRecordingAssembly.

    Final shape:
        dims = ('time_bin', 'neuroid', 'presentation')
        coords:
            time_bin:    time_bin_start_ms, time_bin_end_ms
            neuroid:     neuroid_id, hemisphere, vertex_idx
            presentation: stimulus_id, repetition, subject, run_id
    """
    from brainscore_core.supported_data_standards.brainio.assemblies import (
        NeuronRecordingAssembly,
    )

    # Concatenate along the trial (presentation) axis
    all_windows = []
    all_meta = []
    for subj, windows, meta in subject_data:
        all_windows.append(windows)                       # (n_trials_i, n_tr, n_vertices)
        meta = meta.assign(subject=subj)
        all_meta.append(meta)
    concat_windows = np.concatenate(all_windows, axis=0)  # (n_trials_total, n_tr, n_vertices)
    concat_meta = pd.concat(all_meta, ignore_index=True)

    # We need (time_bin, neuroid, presentation) — transpose
    arr = concat_windows.transpose(1, 2, 0)               # (n_tr, n_vertices, n_trials_total)

    # Build coords
    tr_ms = tr_sec * 1000.0
    time_bin_start_ms = (np.arange(n_tr_per_window) * tr_ms).astype(float)
    time_bin_end_ms   = time_bin_start_ms + tr_ms

    neuroid_id   = [f'fsaverage5.lh.{i}' for i in range(N_VERTICES_PER_HEMI)] + \
                   [f'fsaverage5.rh.{i}' for i in range(N_VERTICES_PER_HEMI)]
    hemisphere   = ['L'] * N_VERTICES_PER_HEMI + ['R'] * N_VERTICES_PER_HEMI
    vertex_idx   = list(range(N_VERTICES_PER_HEMI)) * 2

    assembly = NeuronRecordingAssembly(
        arr,
        coords={
            'time_bin_start_ms': ('time_bin', time_bin_start_ms),
            'time_bin_end_ms':   ('time_bin', time_bin_end_ms),
            'neuroid_id':        ('neuroid', neuroid_id),
            'hemisphere':        ('neuroid', hemisphere),
            'vertex_idx':        ('neuroid', vertex_idx),
            'stimulus_id':       ('presentation', concat_meta['stimulus_id'].values),
            'repetition':        ('presentation', concat_meta['repetition'].values),
            'subject':           ('presentation', concat_meta['subject'].values),
            'run_id':            ('presentation', concat_meta['run_id'].values),
        },
        dims=['time_bin', 'neuroid', 'presentation'],
    )
    return assembly


# ── Stage 4: hashing + upload ─────────────────────────────────────────

def compute_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def upload_to_s3(local_path: Path, bucket: str, key: str) -> str:
    """Upload, return the new version_id."""
    import boto3
    # The 'bucket' arg looks like 'brainscore-storage/brainscore-vision/benchmarks/Lahner2024-fMRI'
    # — split into actual bucket + key prefix.
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
