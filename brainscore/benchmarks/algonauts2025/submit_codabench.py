"""Format Algonauts predictions for Codabench submission.

Codabench expects, per subject, one prediction array per movie split with shape
``(n_TRs, 1000_parcels)``. A submission .zip bundles all subjects' predictions
under a documented layout: ``sub-<NN>/<split>.npy`` plus a ``manifest.json``.

Usage on EC2 (after running ``Algonauts2025FriendsS7.generate_predictions`` per
subject, which writes ``sub-<NN>_<split>.npy`` into a predictions dir):

    python -m brainscore.benchmarks.algonauts2025.submit_codabench \
        --predictions-dir ~/algonauts_predictions/ \
        --output ~/algonauts_friends_s7_submission.zip \
        --split friends_s7

The directory layout follows the challenge's documented per-subject .npy
convention; verify the exact manifest keys against the live Codabench bundle
before the first real submission.
"""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

SCHAEFER_N_PARCELS = 1000


def build_submission(predictions_dir, output, split, subjects=(1, 2, 3, 5)):
    """Bundle per-subject ``sub-<NN>_<split>.npy`` files into a Codabench .zip.

    Each input array must be ``(n_TRs, 1000)``. Returns the manifest dict that
    is also written into the zip as ``manifest.json``. Raises
    ``FileNotFoundError`` for a missing subject file and ``ValueError`` for a
    wrong parcel count.
    """
    predictions_dir = Path(predictions_dir)
    output = Path(output)
    manifest = {'split': split, 'n_parcels': SCHAEFER_N_PARCELS, 'subjects': {}}
    arrays = {}
    for sub in subjects:
        npy = predictions_dir / f'sub-{sub:02d}_{split}.npy'
        if not npy.exists():
            raise FileNotFoundError(
                f"missing prediction for subject {sub}: {npy}")
        arr = np.load(npy)
        if arr.ndim != 2 or arr.shape[1] != SCHAEFER_N_PARCELS:
            raise ValueError(
                f"sub-{sub:02d} prediction must be (n_TRs, {SCHAEFER_N_PARCELS}); "
                f"got {arr.shape}")
        arrays[sub] = arr.astype(np.float32)
        manifest['subjects'][f'sub-{sub:02d}'] = {
            'n_TRs': int(arr.shape[0]), 'path': f'sub-{sub:02d}/{split}.npy'}

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        for sub, arr in arrays.items():
            tmp = output.parent / f'_sub-{sub:02d}_{split}.npy'
            np.save(tmp, arr)
            zf.write(tmp, arcname=f'sub-{sub:02d}/{split}.npy')
            tmp.unlink()
        zf.writestr('manifest.json', json.dumps(manifest, indent=2))
    return manifest


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
    manifest = build_submission(args.predictions_dir, args.output,
                                args.split, tuple(args.subjects))
    print(json.dumps(manifest, indent=2))
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
