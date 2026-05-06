"""Format Algonauts predictions for Codabench submission.

Codabench expects, per subject, one .npy file per movie split with
shape (n_TRs, 1000_parcels). A submission .zip bundles all subjects'
predictions in a specific directory layout.

Usage on EC2 (after running the held-out-prediction benchmark):

    python -m brainscore.benchmarks.algonauts2025.submit_codabench \
        --predictions-dir ~/algonauts_predictions/ \
        --output ~/algonauts_friends_s7_submission.zip \
        --split friends_s7

NOT YET IMPLEMENTED. Sketch only — final layout requires looking at
the Codabench bundle's expected manifest.
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions-dir', type=Path, required=True,
                        help='Directory containing per-(subject, split) '
                             'prediction .npy files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Path to write the submission .zip')
    parser.add_argument('--split', choices=('friends_s7', 'ood'),
                        required=True)
    parser.add_argument('--subjects', type=int, nargs='+',
                        default=[1, 2, 3, 5])
    args = parser.parse_args()

    raise NotImplementedError(
        "submit_codabench — implement once we have a successful test "
        "submission and know the expected zip layout.")


if __name__ == '__main__':
    main()
