"""Unit tests for the Algonauts held-out prediction path + Codabench bundler.

The heavy feature extraction + real ridge fit run on EC2 (need the downloaded
stimuli/assembly). These cover the two data-free pieces: the ridge encoder math
and the submission .zip layout.
"""
import json
import zipfile

import numpy as np
import pytest

from brainscore.benchmarks.algonauts2025.benchmark import fit_predict_ridge
from brainscore.benchmarks.algonauts2025.submit_codabench import (
    build_submission, SCHAEFER_N_PARCELS)


class TestFitPredictRidge:
    def test_recovers_linear_map(self):
        rng = np.random.RandomState(0)
        X = rng.randn(300, 12)
        W = rng.randn(12, 5)
        Y = X @ W + 0.01 * rng.randn(300, 5)
        Xp = rng.randn(50, 12)
        preds = fit_predict_ridge(X, Y, Xp, alpha=1e-3)
        truth = Xp @ W
        # near-perfect recovery with low ridge + low noise
        r = np.corrcoef(preds.ravel(), truth.ravel())[0, 1]
        assert preds.shape == (50, 5)
        assert preds.dtype == np.float32
        assert r > 0.99

    def test_alpha_shrinks(self):
        rng = np.random.RandomState(1)
        X = rng.randn(80, 6)
        Y = rng.randn(80, 3)
        big = fit_predict_ridge(X, Y, X, alpha=1e6)
        # heavy ridge → predictions shrink toward the mean (near-zero variance)
        assert np.var(big) < np.var(Y)


class TestBuildSubmission:
    def _write(self, d, split, subs, n_tr=8):
        for s in subs:
            np.save(d / f'sub-{s:02d}_{split}.npy',
                    np.random.RandomState(s).randn(n_tr, SCHAEFER_N_PARCELS).astype('float32'))

    def test_bundles_zip_with_layout_and_manifest(self, tmp_path):
        self._write(tmp_path, 'friends_s7', [1, 2])
        out = tmp_path / 'sub.zip'
        manifest = build_submission(tmp_path, out, 'friends_s7', subjects=(1, 2))
        assert out.exists()
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
            assert 'sub-01/friends_s7.npy' in names
            assert 'sub-02/friends_s7.npy' in names
            assert 'manifest.json' in names
            m = json.loads(zf.read('manifest.json'))
        assert m['split'] == 'friends_s7'
        assert m['n_parcels'] == SCHAEFER_N_PARCELS
        assert m['subjects']['sub-01']['n_TRs'] == 8

    def test_missing_subject_raises(self, tmp_path):
        self._write(tmp_path, 'ood', [1])
        with pytest.raises(FileNotFoundError, match='subject 2'):
            build_submission(tmp_path, tmp_path / 'o.zip', 'ood', subjects=(1, 2))

    def test_wrong_parcel_count_raises(self, tmp_path):
        np.save(tmp_path / 'sub-01_ood.npy', np.zeros((5, 999), dtype='float32'))
        with pytest.raises(ValueError, match='1000'):
            build_submission(tmp_path, tmp_path / 'o.zip', 'ood', subjects=(1,))
