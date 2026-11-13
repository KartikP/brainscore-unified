"""Held-out predictions for the Algonauts 2025 challenge (Codabench).

Friends season 7 and the out-of-distribution films have no released fMRI, so no
benchmark scores them. ``generate_predictions`` writes a subject's predicted
parcel time series for one of them, for a challenge submission.
"""
import os

import numpy as np

from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025
from brainscore.benchmarks.algonauts2025.scoring import (
    PENALTY_PROTOCOL, PROJECTION_PROTOCOL, fit_predict_banded)


def generate_predictions(subject, held_out: Algonauts2025, out_dir, train_benchmark=None):
    """Predict every scan of ``held_out`` and write ``sub-<NN>_<split>.npy`` into ``out_dir``.

    ``held_out`` is ``Algonauts2025(participant, split='friends_s7' or 'ood', ...)``.
    The method is the scoring method without the outer folds. The bands are
    recorded on the participant's Friends training films and on the held-out
    films. The preparation of each band is fitted on the training films, the
    penalties are chosen on held-out training films, one banded ridge is fitted
    on all training films, and it predicts every scan of ``held_out``. The file
    has shape ``(n_TRs, 1000)``.

    ``train_benchmark`` must be the Friends training split of the same
    participant, with the same channels, stimulus window, HRF delay and
    streaming; by default it is built from ``held_out``. Its ``movies`` limit the
    training films; the ``movies`` of ``held_out`` limit the predicted films.
    """
    if held_out._split == 'friends':
        raise ValueError(
            "generate_predictions is for held-out splits; the friends "
            "training split is scored via __call__.")
    train = train_benchmark or Algonauts2025(
        participant=held_out._participant, channels=held_out._channels,
        streaming=held_out._streaming)
    if train._participant != held_out._participant or train._split != 'friends':
        raise ValueError('Prediction requires Friends training data for the same participant.')
    if train._channels != held_out._channels:
        raise ValueError('Training and prediction must send the same channels.')
    if train._streaming != held_out._streaming:
        raise ValueError('Training and prediction must present the table the same way '
                         '(streaming).')
    if (train._stimulus_window, train._hrf_delay) != (
            held_out._stimulus_window, held_out._hrf_delay):
        raise ValueError('Training and prediction must use the same stimulus window and HRF delay.')
    from brainscore_core.compatibility import check_compatibility
    check_compatibility(subject, held_out)
    train_bands, Y, groups, keep, stack = train.regression_inputs(subject)
    predict_bands, _, _, _, predict_stack = held_out.regression_inputs(subject, drop_excluded=False)
    preds, summary = fit_predict_banded(
        train_bands, Y, groups, keep, stack, predict_bands, predict_stack,
        dim_cap=held_out.LAYER_DIM_CAP, alpha_grid=held_out.BANDED_ALPHA_GRID,
        n_candidates=held_out.BANDED_CANDIDATES)
    os.makedirs(out_dir, exist_ok=True)
    participant, split = held_out._participant, held_out._split
    path = os.path.join(out_dir, f'sub-{participant:02d}_{split}.npy')
    np.save(path, preds)
    return {'path': path, 'predictions': preds, 'participant': participant,
            'split': split, 'n_TRs': int(preds.shape[0]),
            'bands': list(held_out.CHANNEL_REGIONS[held_out._channels]),
            'penalties': summary, 'penalty_protocol': PENALTY_PROTOCOL,
            'feature_projection_protocol': PROJECTION_PROTOCOL}
