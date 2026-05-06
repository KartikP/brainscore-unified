"""One-time preparation: convert Courtois NeuroMod .h5 fMRI files into
brainio-compatible NeuroidAssembly netCDF files.

RUN ON EC2 ONLY. Reads ~100 GB from the DataLad-installed Algonauts
dataset; writes ~25 GB of per-(subject, split) .nc files plus stimulus
metadata CSVs.

Usage on EC2:
    python -m brainscore.benchmarks.algonauts2025.prepare_assembly \
        --algonauts-root ~/algonauts_2025 \
        --output-root ~/.brainio/algonauts2025

Output layout:
    {output_root}/
      algonauts2025_friends_sub01.nc          # (time_bin, neuroid)
      algonauts2025_friends_sub02.nc
      algonauts2025_friends_sub03.nc
      algonauts2025_friends_sub05.nc
      algonauts2025_movie10_sub01.nc          # rolled into 'friends' for now
      algonauts2025_friends_s7_sub01.nc       # NO fMRI ground truth -
                                              #   just shape stub for predictions
      algonauts2025_ood_sub01.nc              # NO fMRI ground truth
      algonauts2025_stim_friends.csv          # train metadata
      algonauts2025_stim_friends_s7.csv       # held-out test metadata
      algonauts2025_stim_ood.csv              # held-out OOD metadata
      algonauts2025_transcripts/<split>/<run>.tsv   # word-onset sidecar

Notes:
- Train assembly merges Friends S1-S6 + Movie10 into one big
  (n_train_TRs, 1000_parcels) per subject.
- Held-out assemblies (S7, OOD) carry only stimulus metadata + the
  expected n_TRs per movie split (from
  ``sub-0X_friends-s7_fmri_samples.npy``).
- Schaefer 1000-parcel atlases per subject are saved alongside as
  ``schaefer1000_sub0X.nii.gz`` for downstream visualization /
  network grouping. Not required for the encoding model itself.

This script is intentionally idempotent: skips any output already
present unless ``--overwrite`` is set.

NOT YET IMPLEMENTED — file structure stub. Sketch of the work below.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


# Subjects in the Algonauts release (per the challenge README)
ALGONAUTS_SUBJECTS = (1, 2, 3, 5)

# Splits (one per held-out / training partition)
ALGONAUTS_SPLITS = ('friends', 'friends_s7', 'ood')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algonauts-root', type=Path, required=True,
                        help='DataLad-installed dataset directory '
                             '(produced by download_algonauts_data.sh)')
    parser.add_argument('--output-root', type=Path,
                        default=Path('~/.brainio/algonauts2025').expanduser())
    parser.add_argument('--subjects', type=int, nargs='+',
                        default=list(ALGONAUTS_SUBJECTS))
    parser.add_argument('--splits', type=str, nargs='+',
                        default=list(ALGONAUTS_SPLITS),
                        choices=ALGONAUTS_SPLITS)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f'Algonauts root:  {args.algonauts_root}')
    print(f'Output root:     {args.output_root}')
    print(f'Subjects:        {args.subjects}')
    print(f'Splits:          {args.splits}')

    # 1. For each split, build the StimulusSet CSV listing every movie
    #    segment + path to .mkv + path to .tsv transcript + expected
    #    sample count.
    for split in args.splits:
        print(f'\n--- Stimulus_set for {split} ---')
        df = _build_stim_set_csv(args.algonauts_root, split)
        out = args.output_root / f'algonauts2025_stim_{split}.csv'
        if out.exists() and not args.overwrite:
            print(f'  Skipping (exists): {out}')
        else:
            df.to_csv(out, index=False)
            print(f'  Wrote {len(df)} rows → {out}')

    # 2. For each (split, subject), build the (time_bin, neuroid)
    #    NeuralAssembly netCDF.
    for split in args.splits:
        for subject in args.subjects:
            out = (args.output_root /
                   f'algonauts2025_{split}_sub{subject:02d}.nc')
            if out.exists() and not args.overwrite:
                print(f'\n--- Skipping (exists): {out}')
                continue
            print(f'\n--- Assembly for {split} sub-{subject:02d} ---')
            if split == 'friends':
                # Train split: merge Friends S1-S6 + Movie10 .h5 files
                _build_train_assembly(
                    args.algonauts_root, subject, out)
            else:
                # Held-out: build a stub with the expected shape only
                _build_held_out_stub(
                    args.algonauts_root, subject, split, out)
    print('\nDone.')


def _build_stim_set_csv(algonauts_root: Path, split: str):
    """Walk stimuli/<split> + transcripts/<split> and produce a DataFrame
    with one row per (movie, episode, episode_split) chunk.

    Columns: stimulus_id, movie, episode, episode_split, run, video_path,
    transcript_path, expected_n_TRs.
    """
    import pandas as pd
    raise NotImplementedError(
        f"_build_stim_set_csv({split!r}) — implement once the file "
        f"layout is verified on EC2.")


def _build_train_assembly(algonauts_root: Path, subject: int,
                          output_path: Path):
    """Merge Friends S1-S6 + Movie10 .h5 files for one subject into a
    single (time_bin, neuroid=1000) NeuroidAssembly netCDF.

    Expected input layout:
        algonauts_root/fmri/sub-0X/func/<subject>_task-friends_*.h5
        algonauts_root/fmri/sub-0X/func/<subject>_task-movie10_*.h5

    Each .h5 contains datasets keyed by movie segment. We:
    1. Iterate datasets, concat along time_bin
    2. Tag each sample with (movie, episode, split, run, sample_idx)
    3. Save as xarray netCDF (presentation, neuroid) with
       MultiIndex over (subject, movie, episode, episode_split, run, t)
    """
    raise NotImplementedError(
        "_build_train_assembly — implement once we've verified the .h5 "
        "structure on EC2.")


def _build_held_out_stub(algonauts_root: Path, subject: int, split: str,
                          output_path: Path):
    """For held-out splits (Friends S7, OOD), no fMRI ground truth exists.
    We still need a benchmark target shape so the model can write its
    predictions in the correct (n_TRs, 1000) format.

    The challenge ships per-subject .npy files with the expected number
    of TRs per movie segment (in ``target_samples_number/``). Read those,
    write a stub assembly with NaN values of the right shape — the
    benchmark will replace these with the model's predictions before
    saving the Codabench submission.
    """
    raise NotImplementedError(
        "_build_held_out_stub — implement once stimulus + sample-count "
        "layouts are verified on EC2.")


if __name__ == '__main__':
    main()
