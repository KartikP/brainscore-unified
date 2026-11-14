"""Direct-comparison metric for whole-model→whole-brain encoders.

Conventional Brain-Score metrics fit a readout (PLS / ridge) from model
features to neural data. A *trained brain encoder* (e.g. TRIBEv2) instead
outputs brain space directly — one predicted value per recorded unit — so the
right evaluation is a direct per-unit correlation of predicted-vs-measured
responses across stimuli, with NO fitted readout.

``DirectComparisonMetric`` aligns a prediction assembly and a measured
assembly by ``stimulus_id`` (averaging measured reps per stimulus), then
reports the per-unit Pearson r across stimuli plus a clip-shuffle null
(permute the prediction's stimulus order) — the null should sit near zero,
confirming the correspondence is real and the unit ordering is aligned. The
prediction and target must share unit ordering (both, e.g., fsaverage5
vertices in canonical order); the metric checks the counts match.
"""
import numpy as np

from brainscore_core.metrics import Metric, Score, per_unit_pearson


def _stimulus_ids(a):
    """Read stimulus_id whether it is a top-level coord (loaded assemblies),
    a 'presentation' MultiIndex level, or an attribute-exposed coord (brainio
    collapses a single presentation coord into a 'presentation'-named index)."""
    if 'stimulus_id' in a.coords:
        return np.asarray(a['stimulus_id'].values)
    idx = a.indexes.get('presentation') if hasattr(a, 'indexes') else None
    if idx is not None and 'stimulus_id' in (getattr(idx, 'names', None) or []):
        return np.asarray(idx.get_level_values('stimulus_id'))
    if hasattr(a, 'stimulus_id'):
        return np.asarray(a.stimulus_id.values)
    raise KeyError("no stimulus_id on the presentation axis")


def _extract(assembly):
    """Return ``(data (presentation, neuroid), stimulus_ids)``."""
    a = assembly.transpose('presentation', 'neuroid')
    return np.asarray(a.values, dtype='float64'), _stimulus_ids(a)


class DirectComparisonMetric(Metric):
    """Per-unit Pearson of predicted vs measured responses + clip-shuffle null.

    ``__call__(prediction, target) -> Score`` where ``Score`` is the median
    per-unit r; ``score.attrs`` carries ``raw``/``null``/``mean_r``/
    ``top10pct_r``/``frac_units_r_gt_0.1``/``n_stimuli``/``n_units``.
    """

    def __init__(self, shuffle_seed: int = 0):
        self._shuffle_seed = shuffle_seed

    def _align(self, X, x_ids, Y_full, y_ids):
        """Mean measured reps per stimulus, aligned to the prediction's
        stimulus order; drop prediction stimuli with no measured match."""
        y_ids = np.asarray(y_ids)
        matched_X, matched_Y = [], []
        for i, sid in enumerate(x_ids):
            m = y_ids == sid
            if m.sum() == 0:
                continue
            matched_X.append(X[i])
            matched_Y.append(Y_full[m].mean(axis=0))
        if not matched_X:
            raise ValueError(
                "no shared stimulus_id between prediction and target.")
        return np.stack(matched_X), np.stack(matched_Y)

    def __call__(self, prediction, target) -> Score:
        X, x_ids = _extract(prediction)
        Y_full, y_ids = _extract(target)
        if X.shape[1] != Y_full.shape[1]:
            raise ValueError(
                f"prediction has {X.shape[1]} units but target has "
                f"{Y_full.shape[1]} — they must share unit ordering.")
        X, Y = self._align(X, x_ids, Y_full, y_ids)

        r = per_unit_pearson(X, Y)
        perm = np.random.RandomState(self._shuffle_seed).permutation(len(X))
        r_null = per_unit_pearson(X[perm], Y)

        finite = r[np.isfinite(r)]
        top10 = (float(np.median(np.sort(finite)[-max(1, int(len(finite) * 0.1)):]))
                 if finite.size else float('nan'))
        median_r = float(np.nanmedian(r))

        score = Score(median_r)
        score.attrs['raw'] = median_r
        score.attrs['null'] = float(np.nanmedian(r_null))
        score.attrs['mean_r'] = float(np.nanmean(r))
        score.attrs['top10pct_r'] = top10
        score.attrs['frac_units_r_gt_0.1'] = float(np.nanmean(r > 0.1))
        score.attrs['n_stimuli'] = int(len(X))
        score.attrs['n_units'] = int(X.shape[1])
        return score
