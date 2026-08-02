"""Unit tests for the Algonauts held-out prediction path + Codabench bundler.

The heavy feature extraction + real ridge fit run on EC2 (need the downloaded
stimuli/assembly). These cover the two data-free pieces: the ridge encoder math
and the submission .zip layout.
"""
import zipfile

import numpy as np
import pytest

from brainscore.tools.banded_ridge import ridge_fit_predict

# The Codabench submission bundler is an operational driver, not library code — it
# deliberately does not ship (see test_data_plugins.py, which forbids it inside the
# benchmark package). Skip its tests rather than fail collection where it is absent.
submit_codabench = pytest.importorskip(
    'experiments.algonauts2025.submit_codabench',
    reason='submission bundler is a local operational driver, not part of the package')
build_submission = submit_codabench.build_submission
write_submission_zip = submit_codabench.write_submission_zip
SCHAEFER_N_PARCELS = submit_codabench.SCHAEFER_N_PARCELS
SPLIT_NPY_NAME = submit_codabench.SPLIT_NPY_NAME


class TestFitPredictRidge:
    def test_recovers_linear_map(self):
        rng = np.random.RandomState(0)
        X = rng.randn(300, 12)
        W = rng.randn(12, 5)
        Y = X @ W + 0.01 * rng.randn(300, 5)
        Xp = rng.randn(50, 12)
        preds = ridge_fit_predict(X, Y, Xp, alpha=1e-3)
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
        big = ridge_fit_predict(X, Y, X, alpha=1e6)
        # heavy ridge → predictions shrink toward the mean (near-zero variance)
        assert np.var(big) < np.var(Y)


class TestWriteSubmissionZip:
    """Core formatter: in-memory nested dict -> Codabench zip (nested .npy)."""

    def _nested(self, subs, episodes, n_tr=8):
        return {
            f'sub-{s:02d}': {
                epi: np.random.RandomState(s * 10 + i).randn(
                    n_tr + i, SCHAEFER_N_PARCELS).astype('float32')
                for i, epi in enumerate(episodes)}
            for s in subs}

    def test_zip_contains_single_nested_npy(self, tmp_path):
        nested = self._nested([1, 2, 3, 5],
                              ['friends_s07e01a', 'friends_s07e01b'])
        out = tmp_path / 'submission.zip'
        write_submission_zip(nested, out, 'friends_s7')
        assert out.exists()
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert names == [SPLIT_NPY_NAME['friends_s7']]  # exactly one .npy
            zf.extractall(tmp_path / 'x')
        loaded = np.load(
            tmp_path / 'x' / SPLIT_NPY_NAME['friends_s7'],
            allow_pickle=True).item()
        # round-trips the nested structure: {sub: {episode: (n_TRs, 1000)}}
        assert set(loaded) == {'sub-01', 'sub-02', 'sub-03', 'sub-05'}
        assert set(loaded['sub-01']) == {'friends_s07e01a', 'friends_s07e01b'}
        arr = loaded['sub-01']['friends_s07e01a']
        assert arr.shape == (8, SCHAEFER_N_PARCELS)
        assert arr.dtype == np.float32

    def test_ood_split_uses_ood_npy_name(self, tmp_path):
        nested = self._nested([1], ['chaplin1', 'mononoke'])
        out = tmp_path / 'ood.zip'
        write_submission_zip(nested, out, 'ood')
        with zipfile.ZipFile(out) as zf:
            assert zf.namelist() == [SPLIT_NPY_NAME['ood']]

    def test_unknown_split_raises(self, tmp_path):
        with pytest.raises(ValueError, match='split must be one of'):
            write_submission_zip(self._nested([1], ['e1']),
                                 tmp_path / 'x.zip', 'friends_s99')

    def test_wrong_parcel_count_raises(self, tmp_path):
        bad = {'sub-01': {'e1': np.zeros((5, 999), dtype='float32')}}
        with pytest.raises(ValueError, match='1000'):
            write_submission_zip(bad, tmp_path / 'x.zip', 'ood')

    def test_empty_episode_dict_raises(self, tmp_path):
        with pytest.raises(ValueError, match='empty'):
            write_submission_zip({'sub-01': {}}, tmp_path / 'x.zip', 'ood')


class TestBuildSubmission:
    """Assembles per-subject episode-dict .npy files into the nested zip."""

    def _write_subject(self, d, split, sub, episodes, n_tr=8):
        episode_dict = {
            epi: np.random.RandomState(sub * 10 + i).randn(
                n_tr, SCHAEFER_N_PARCELS).astype('float32')
            for i, epi in enumerate(episodes)}
        np.save(d / f'sub-{sub:02d}_{split}.npy', episode_dict,
                allow_pickle=True)

    def test_assembles_subjects_into_nested_zip(self, tmp_path):
        for s in (1, 2):
            self._write_subject(tmp_path, 'friends_s7', s,
                                ['friends_s07e01a', 'friends_s07e02a'])
        out = tmp_path / 'sub.zip'
        nested = build_submission(tmp_path, out, 'friends_s7', subjects=(1, 2))
        assert set(nested) == {'sub-01', 'sub-02'}
        assert set(nested['sub-01']) == {'friends_s07e01a', 'friends_s07e02a'}
        with zipfile.ZipFile(out) as zf:
            assert zf.namelist() == [SPLIT_NPY_NAME['friends_s7']]

    def test_missing_subject_raises(self, tmp_path):
        self._write_subject(tmp_path, 'ood', 1, ['chaplin1'])
        with pytest.raises(FileNotFoundError, match='sub-02'):
            build_submission(tmp_path, tmp_path / 'o.zip', 'ood', subjects=(1, 2))

    def test_wrong_parcel_count_raises(self, tmp_path):
        np.save(tmp_path / 'sub-01_ood.npy',
                {'chaplin1': np.zeros((5, 999), dtype='float32')},
                allow_pickle=True)
        with pytest.raises(ValueError, match='1000'):
            build_submission(tmp_path, tmp_path / 'o.zip', 'ood', subjects=(1,))
