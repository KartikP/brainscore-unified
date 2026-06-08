"""Format Algonauts 2025 predictions for the Codabench submission.

The Codabench scorer expects a single pickled ``.npy`` holding a NESTED dict::

    { 'sub-01': { '<episode>': np.ndarray(n_TRs, 1000) float32, ... },
      'sub-02': {...}, 'sub-03': {...}, 'sub-05': {...} }

zipped with the ``.npy`` stored by basename. Episode keys are the per-episode
names from ``target_sample_number/sub-0X_{friends-s7,ood}_fmri_samples.npy``
(e.g. ``friends_s07e01a`` for S7, ``chaplin1`` / ``mononoke`` for OOD), and each
array's row count must equal that episode's recorded fMRI sample count. This
matches the official dev-kit notebook (cells 117 / 132); verified 2026-06-08.

Per-subject prediction files (written by ``generate_predictions``) are
``sub-0X_<split>.npy``, each a pickled per-subject episode dict
``{episode: (n_TRs, 1000) float32}``. ``build_submission`` assembles the four
subjects into the nested-dict zip.

Usage::

    python -m brainscore.benchmarks.algonauts2025.submit_codabench \
        --predictions-dir ~/algonauts_predictions/ \
        --output ~/fmri_predictions_friends_s7.zip \
        --split friends_s7
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np

SCHAEFER_N_PARCELS = 1000
SUBJECTS = (1, 2, 3, 5)

# Codabench expects these exact .npy basenames inside the submission zip.
SPLIT_NPY_NAME = {
    'friends_s7': 'fmri_predictions_friends_s7.npy',
    'ood': 'fmri_predictions_ood.npy',
}


def _validate_episode_dict(sub_label, episode_dict):
    if not isinstance(episode_dict, dict):
        raise ValueError(
            f"{sub_label}: expected a {{episode: array}} dict, got "
            f"{type(episode_dict).__name__}")
    if not episode_dict:
        raise ValueError(f"{sub_label}: episode dict is empty")
    for epi, arr in episode_dict.items():
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != SCHAEFER_N_PARCELS:
            raise ValueError(
                f"{sub_label}/{epi}: prediction must be (n_TRs, "
                f"{SCHAEFER_N_PARCELS}); got {arr.shape}")


def write_submission_zip(nested_predictions, output, split):
    """Write a Codabench submission zip from an in-memory nested dict
    ``{sub_label: {episode: (n_TRs, 1000)}}``.

    Validates parcel count per episode, coerces arrays to float32, pickles the
    nested dict via ``np.save`` and zips it under the Codabench-expected
    basename. Returns the zip ``Path``.
    """
    if split not in SPLIT_NPY_NAME:
        raise ValueError(
            f"split must be one of {list(SPLIT_NPY_NAME)}; got {split!r}")
    coerced = {}
    for sub_label, episode_dict in nested_predictions.items():
        _validate_episode_dict(sub_label, episode_dict)
        coerced[sub_label] = {
            epi: np.asarray(a, dtype=np.float32)
            for epi, a in episode_dict.items()}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    npy_name = SPLIT_NPY_NAME[split]
    tmp_npy = output.parent / npy_name
    np.save(tmp_npy, coerced, allow_pickle=True)
    try:
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(tmp_npy, arcname=npy_name)
    finally:
        tmp_npy.unlink()
    return output


def build_submission(predictions_dir, output, split, subjects=SUBJECTS):
    """Assemble per-subject ``sub-0X_<split>.npy`` episode-dict files into the
    nested-dict Codabench zip.

    Each per-subject file is a pickled ``{episode: (n_TRs, 1000)}`` dict.
    Returns the assembled nested prediction dict. Raises ``FileNotFoundError``
    for a missing subject file and ``ValueError`` for a wrong parcel count.
    """
    predictions_dir = Path(predictions_dir)
    nested = {}
    for sub in subjects:
        sub_label = f'sub-{sub:02d}'
        npy = predictions_dir / f'{sub_label}_{split}.npy'
        if not npy.exists():
            raise FileNotFoundError(
                f"missing prediction for {sub_label}: {npy}")
        nested[sub_label] = np.load(npy, allow_pickle=True).item()
    write_submission_zip(nested, output, split)
    return nested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions-dir', type=Path, required=True,
                        help='Directory containing per-subject '
                             'sub-0X_<split>.npy episode-dict files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Path to write the submission .zip')
    parser.add_argument('--split', choices=tuple(SPLIT_NPY_NAME),
                        required=True)
    parser.add_argument('--subjects', type=int, nargs='+',
                        default=list(SUBJECTS))
    args = parser.parse_args()
    nested = build_submission(args.predictions_dir, args.output,
                              args.split, tuple(args.subjects))
    for sub_label, episodes in nested.items():
        print(f'{sub_label}: {len(episodes)} episodes, '
              f'shapes {[tuple(np.asarray(a).shape) for a in episodes.values()][:3]}...')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
