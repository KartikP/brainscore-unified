"""Whole-cortex voxelwise encoding on LeBel 2023 story-listening fMRI.

One subject (UTS03) heard 25 spoken stories while BOLD was recorded every 2 s
across 20484 cortical vertices. The benchmark asks how much of that signal a
model's language representation can predict.

Protocol
--------
Text features are read from the candidate at one sample per TR, then delayed by
1-4 TRs and concatenated. The delays let a linear model learn its own
hemodynamic response per vertex, which is the convention in this literature and
avoids committing to a fixed HRF shape.

Cross-validation holds out whole stories. Neighbouring TRs within a story are
strongly autocorrelated, so a random split over samples would place near-copies
of a test sample in the training set and inflate the score.

The reported value is the median Pearson r across vertices, held out. It is a
raw correlation: the stimuli are heard once each, so there are no repeats from
which to estimate a noise ceiling, and no normalisation is applied.
"""

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score

from .._scoring_utils import pearson_summary, run_kfold_masks
from ...tools.banded_ridge import (
    _mean_pearson, dual_aware_ridge_predict, solve_and_project)

BIBTEX = """@article{lebel2023natural,
  title={A natural language fMRI dataset for voxelwise encoding models},
  author={LeBel, Amanda and Wagner, Lauren and Jain, Shailee and Adhikari-Desai,
          Aneesh and Gupta, Bhavin and Morgenthal, Allyson and Tang, Jerry and
          Xu, Lixiang and Huth, Alexander G},
  journal={Scientific Data},
  volume={10}, number={1}, pages={555}, year={2023},
  publisher={Nature Publishing Group}
}"""

DEFAULT_DELAYS = (1, 2, 3, 4)
DEFAULT_ALPHA_GRID = (1e4, 1e5, 1e6, 3e6, 1e7, 3e7, 1e8)


def _delay_within_stories(features, story_ids, delays):
    """Concatenate copies of ``features`` shifted back by each delay.

    Shifting happens inside each story, so the opening TRs of one story are
    never predicted from the closing TRs of the previous one.

    Returns ``(delayed, has_full_history)``. The opening TRs of a story cannot
    be filled for every delay, and padding them is not safe: the padding is
    identical across stories and lines up with the large BOLD response to a
    story starting, so a model that emits a constant vector can "predict" it.
    Measured, that artifact was worth a median r of 0.054 — comparable to the
    real effect being measured. Those rows are therefore excluded rather than
    padded.
    """
    n_samples, n_features = features.shape
    max_delay = max(delays)

    # Which rows survive is known from the story layout alone, so the design
    # matrix is allocated at its final size and filled directly. Building it
    # full-length and then indexing would hold two copies at once, and the wide
    # ones are large: nine layers of a 27B model reached 35 GB that way.
    has_full_history = np.zeros(n_samples, dtype=bool)
    story_rows = {}
    for story in np.unique(story_ids):
        rows = np.flatnonzero(story_ids == story)
        story_rows[story] = rows
        has_full_history[rows[max_delay:]] = True
    kept = np.flatnonzero(has_full_history)

    out = np.zeros((len(kept), n_features * len(delays)), dtype=np.float32)
    for story, rows in story_rows.items():
        block = features[rows]
        # Where this story's surviving rows land in the compacted output.
        positions = np.searchsorted(kept, rows[max_delay:])
        for d_i, delay in enumerate(delays):
            span = slice(d_i * n_features, (d_i + 1) * n_features)
            out[positions, span] = block[max_delay - delay:len(rows) - delay]
    return out, has_full_history


def _zscore_per_story(values, story_ids):
    """Standardise every vertex within every story.

    BOLD magnitude is arbitrary and drifts between runs. Without this, a model
    can appear predictive simply by tracking which story is playing.
    """
    out = np.empty_like(values, dtype=np.float32)
    for story in np.unique(story_ids):
        rows = np.flatnonzero(story_ids == story)
        block = values[rows].astype(np.float32)
        block -= block.mean(axis=0)
        std = block.std(axis=0)
        out[rows] = np.divide(block, std, out=np.zeros_like(block), where=std > 0)
    return out


def _ridge_grouped_alpha(X_train, Y_train, X_test, stories_train, alpha_grid,
                         random_state=0):
    """Ridge whose penalty is chosen on held-out *stories*, then refit in full.

    Selecting the penalty on a random subset of rows does not work here: TRs
    within a story are so correlated that near-duplicates of a validation row
    sit in the selection-training set, which makes weak penalties look better
    than they generalise. Holding out whole stories removes that shortcut. The
    effect is large — a random inner split picks a penalty roughly an order of
    magnitude too small.

    Whichever Gram matrix is smaller is computed once and reused across the
    grid, since only the diagonal penalty term changes per candidate.
    """
    stories = np.unique(stories_train)
    rng = np.random.default_rng(random_state)
    n_val = max(1, len(stories) // 4)
    val_stories = rng.choice(stories, n_val, replace=False)
    is_val = np.isin(stories_train, val_stories)

    # Centre features on the selection-training rows only; ridge here has no
    # intercept term, so uncentred features would be penalised for their mean.
    inner_mean = X_train[~is_val].mean(axis=0)
    X_inner = X_train[~is_val] - inner_mean
    X_val = X_train[is_val] - inner_mean
    Y_inner, Y_val = Y_train[~is_val], Y_train[is_val]

    Y_val_centered = Y_val - Y_val.mean(axis=0)
    Y_val_var = (Y_val_centered ** 2).sum(axis=0)

    n_inner, n_features = X_inner.shape
    use_dual = n_features > n_inner
    if use_dual:
        gram = X_inner @ X_inner.T           # (n, n)
        cross = X_val @ X_inner.T            # maps dual weights onto the split
        target = Y_inner
    else:
        gram = X_inner.T @ X_inner           # (p, p)
        target = X_inner.T @ Y_inner

    if not use_dual:
        cross = X_val

    best_alpha, best_score = None, -np.inf
    diagonal = np.diag_indices_from(gram)
    penalised = gram.copy()
    for alpha in alpha_grid:
        penalised[diagonal] = gram[diagonal] + alpha
        predicted = solve_and_project(penalised, cross, target)
        score = _mean_pearson(Y_val_centered, Y_val_var, predicted)
        if score > best_score:
            best_score, best_alpha = score, alpha

    full_mean = X_train.mean(axis=0)
    return dual_aware_ridge_predict(
        X_train - full_mean, Y_train, X_test - full_mean, best_alpha), best_alpha


class LeBel2023Encoding(BenchmarkBase):
    """Voxelwise encoding score over all measured cortical vertices."""

    def __init__(self, identifier='LeBel2023-UTS03-encoding', region='language_system',
                 delays=DEFAULT_DELAYS, alpha_grid=DEFAULT_ALPHA_GRID,
                 n_splits=5, context_window_tr=5, max_targets=None,
                 random_state=0):
        # Ceiling is 1.0 because none is estimable here (see module docstring);
        # the reported value is therefore a raw correlation.
        super().__init__(identifier=identifier, ceiling=Score(1.0),
                         version=1, parent='neural_language', bibtex=BIBTEX)
        self._region = region
        self._delays = tuple(delays)
        self._alpha_grid = tuple(alpha_grid)
        self._n_splits = n_splits
        self._context_window_tr = context_window_tr
        # Subsampling targets keeps a smoke run cheap; None scores all vertices.
        self._max_targets = max_targets
        self._random_state = random_state
        self._cached = None

    def _data(self):
        if self._cached is None:
            from ...data.lebel2023.data import load
            self._cached = load(context_window_tr=self._context_window_tr)
        return self._cached

    def _model_features(self, candidate, stimulus_set, assembly):
        """Drive the candidate and return features in the assembly's row order.

        Isolated from scoring so that features obtained another way — a model
        too large for the scoring environment, or a deliberately uninformative
        null — can be scored through exactly this path.
        """
        candidate.start_recording(self._region, recording_type='fMRI')
        predictions = candidate.process(stimulus_set)
        return self._align_features(predictions, assembly)

    def __call__(self, candidate) -> Score:
        stimulus_set, assembly = self._data()

        features = self._model_features(candidate, stimulus_set, assembly)
        story_ids = np.asarray(assembly['story_id'].values)

        X, has_full_history = _delay_within_stories(
            features, story_ids, self._delays)
        Y = _zscore_per_story(np.asarray(assembly.values), story_ids)
        Y, target_index = self._select_targets(Y)

        # X already arrives compacted to the rows with full delay history;
        # the targets and story labels are cut down to match. Z-scoring above
        # used every TR of the story, so each story is still standardised
        # against its own full time course.
        Y, story_ids = Y[has_full_history], story_ids[has_full_history]

        held_out = np.full(Y.shape, np.nan, dtype=np.float32)
        chosen_alphas = []
        # Whole stories are held out; `run_kfold_masks` already does exactly
        # this grouping for run ids, and a story is the same kind of block.
        for train_mask, test_mask in run_kfold_masks(
                story_ids, self._n_splits, self._random_state):
            fold_prediction, alpha = _ridge_grouped_alpha(
                X[train_mask], Y[train_mask], X[test_mask],
                story_ids[train_mask], self._alpha_grid,
                random_state=self._random_state)
            held_out[test_mask] = fold_prediction
            chosen_alphas.append(float(alpha))

        per_vertex, median_r, mean_r = pearson_summary(Y, held_out)

        score = Score(median_r)
        score.attrs['raw'] = per_vertex
        score.attrs['mean_r'] = mean_r
        score.attrs['n_targets'] = int(Y.shape[1])
        score.attrs['target_index'] = target_index
        score.attrs['alphas'] = chosen_alphas
        score.attrs['ceiling'] = 'none — single presentation per story, raw r reported'
        score.attrs['error'] = float(np.std(per_vertex) / np.sqrt(len(per_vertex)))
        return score

    def _align_features(self, predictions, assembly):
        """Order model features to match the assembly's rows.

        Extraction may reorder or cache rows, so features are matched on
        ``stimulus_id`` rather than trusting position.
        """
        feature_ids = np.asarray(predictions['stimulus_id'].values)
        wanted = np.asarray(assembly['stimulus_id'].values)
        position = {sid: i for i, sid in enumerate(feature_ids)}
        missing = [sid for sid in wanted if sid not in position]
        if missing:
            raise ValueError(
                f"model returned no features for {len(missing)} stimuli "
                f"(first: {missing[0]})")
        order = np.fromiter((position[sid] for sid in wanted), dtype=int,
                            count=len(wanted))
        return np.asarray(predictions.values, dtype=np.float32)[order]

    def _select_targets(self, Y):
        """Optionally score a random subset of vertices, for cheap smoke runs."""
        if self._max_targets is None or self._max_targets >= Y.shape[1]:
            return Y, None
        rng = np.random.default_rng(self._random_state)
        index = np.sort(rng.choice(Y.shape[1], self._max_targets, replace=False))
        return Y[:, index], index
