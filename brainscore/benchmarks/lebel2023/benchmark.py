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

TR_SEC = 2.0
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



def _ridge_per_voxel_alpha(X_train, Y_train, X_test, stories_train, alpha_grid,
                           random_state=0, n_inner_folds=4):
    """Ridge with a penalty chosen independently for every target.

    One shared penalty is a compromise across targets whose signal-to-noise
    ratios differ by orders of magnitude: whole-cortex data mixes reliable
    language-network vertices with vertices carrying almost no stimulus-locked
    signal. Fitting a penalty per target is what reference encoding pipelines do
    (LITcoder's ``single_alpha=False`` is its default).

    The selection must be *nested*, not a single split. Choosing each target's
    penalty on one held-out set means fitting one noisy estimate per target, and
    with thousands of targets that selection noise dominates: measured on this
    benchmark, single-split per-target selection scored 35% *below* a single
    shared penalty. Averaging the validation scores over several inner folds
    before selecting is what makes the extra freedom pay rather than cost.

    Inner folds hold out whole stories, as elsewhere here — a random row split
    leaks across autocorrelated samples.

    Returns ``(predictions, chosen_alphas)`` with one alpha per target.
    """
    from brainscore_core.metrics import per_unit_pearson

    n_targets = Y_train.shape[1]
    val_scores = np.zeros((len(alpha_grid), n_targets), dtype=np.float64)
    folds = 0
    for inner_train, inner_val in run_kfold_masks(
            stories_train, n_splits=n_inner_folds, random_state=random_state):
        if not inner_val.any() or not inner_train.any():
            continue
        folds += 1
        inner_mean = X_train[inner_train].mean(axis=0)
        X_inner = X_train[inner_train] - inner_mean
        X_val = X_train[inner_val] - inner_mean
        Y_inner, Y_val = Y_train[inner_train], Y_train[inner_val]

        n_inner, n_features = X_inner.shape
        use_dual = n_features > n_inner
        if use_dual:
            gram = X_inner @ X_inner.T
            cross, target = X_val @ X_inner.T, Y_inner
        else:
            gram = X_inner.T @ X_inner
            cross, target = X_val, X_inner.T @ Y_inner

        diagonal = np.diag_indices_from(gram)
        penalised = gram.copy()
        for index, alpha in enumerate(alpha_grid):
            penalised[diagonal] = gram[diagonal] + alpha
            predicted = solve_and_project(penalised, cross, target)
            val_scores[index] += np.nan_to_num(
                per_unit_pearson(Y_val, predicted), nan=0.0)
    val_scores /= max(folds, 1)

    chosen = val_scores.argmax(axis=0)
    alphas = np.asarray(alpha_grid, dtype=float)[chosen]

    # Refit on the full training fold, one solve per distinct penalty.
    full_mean = X_train.mean(axis=0)
    X_full, X_test_centred = X_train - full_mean, X_test - full_mean
    predictions = np.empty((X_test.shape[0], n_targets), dtype=np.float32)
    for index, alpha in enumerate(alpha_grid):
        columns = np.flatnonzero(chosen == index)
        if columns.size:
            predictions[:, columns] = dual_aware_ridge_predict(
                X_full, Y_train[:, columns], X_test_centred, alpha)
    return predictions, alphas


class LeBel2023Encoding(BenchmarkBase):
    """Voxelwise encoding score over all measured cortical vertices."""

    def __init__(self, identifier='LeBel2023-UTS03-encoding', region='language_system',
                 delays=DEFAULT_DELAYS, alpha_grid=DEFAULT_ALPHA_GRID,
                 n_splits=5, context_window_tr=5, max_targets=None,
                 per_voxel_alpha=False, random_state=0):
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
        # One penalty per target rather than one shared across all of them.
        self._per_voxel_alpha = per_voxel_alpha
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
            fitter = (_ridge_per_voxel_alpha if self._per_voxel_alpha
                      else _ridge_grouped_alpha)
            fold_prediction, alpha = fitter(
                X[train_mask], Y[train_mask], X[test_mask],
                story_ids[train_mask], self._alpha_grid,
                random_state=self._random_state)
            held_out[test_mask] = fold_prediction
            chosen_alphas.append(
                np.median(alpha).item() if np.ndim(alpha) else float(alpha))

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

class LeBel2023EncodingWordLevel(LeBel2023Encoding):
    """Same scoring, but features are read per word and resampled onto the TRs.

    The parent collapses each TR to one context window up front and reads a
    single vector for it, which discards the several words that fall inside a TR.
    The reference pipeline instead reads a representation per word and then
    downsamples word-time features onto the fMRI grid, and reports that
    last-token selection is its worst-performing aggregation.

    ``pooling`` selects how word-time features become TR-time features:

    - ``lanczos``  windowed-sinc resampling, low-passed at the TR rate
    - ``average``  mean of the words whose onset falls inside the TR
    - ``sum``      sum of those words
    - ``last``     the final word before the TR (what the parent effectively does)
    """

    VALID_POOLING = ('lanczos', 'average', 'sum', 'last')

    def __init__(self, pooling='lanczos', context_words=32, **kwargs):
        if pooling not in self.VALID_POOLING:
            raise ValueError(
                f'pooling must be one of {self.VALID_POOLING}; got {pooling!r}')
        super().__init__(**kwargs)
        self._pooling = pooling
        self._context_words = context_words
        self._word_cached = None

    def _word_data(self):
        if self._word_cached is None:
            from ...data.lebel2023.data import build_word_level
            self._word_cached = build_word_level(
                context_words=self._context_words)
        return self._word_cached

    def _data(self):
        word_stimuli, assembly, _ = self._word_data()
        return word_stimuli, assembly

    def _model_features(self, candidate, stimulus_set, assembly):
        """Read one feature per word, then resample onto the TR grid."""
        from brainscore_core.temporal import lanczos_downsample

        word_stimuli, _, tr_times_by_story = self._word_data()
        candidate.start_recording(self._region, recording_type='fMRI')
        predictions = candidate.process(word_stimuli)

        # Order model output to the word rows, matching on id rather than position.
        position = {sid: i for i, sid in
                    enumerate(np.asarray(predictions['stimulus_id'].values))}
        wanted = np.asarray(word_stimuli['stimulus_id'].values)
        missing = [s for s in wanted if s not in position]
        if missing:
            raise ValueError(f'model returned no features for {len(missing)} '
                             f'words (first: {missing[0]})')
        order = np.fromiter((position[s] for s in wanted), dtype=int,
                            count=len(wanted))
        word_features = np.asarray(predictions.values, dtype=np.float32)[order]

        word_story = np.asarray(word_stimuli['story_id'].values)
        word_time = np.asarray(word_stimuli['word_time_sec'].values, dtype=float)
        assembly_story = np.asarray(assembly['story_id'].values)

        out = np.zeros((len(assembly_story), word_features.shape[1]),
                       dtype=np.float32)
        for story, tr_times in tr_times_by_story.items():
            rows = np.flatnonzero(word_story == story)
            targets = np.flatnonzero(assembly_story == story)
            out[targets] = self._pool(
                word_features[rows], word_time[rows], tr_times,
                lanczos_downsample)
        return out

    def _pool(self, features, feature_times, tr_times, lanczos):
        if self._pooling == 'lanczos':
            return lanczos(features, feature_times, tr_times)
        # The remaining strategies act on the words falling inside each TR.
        edges = np.searchsorted(feature_times, tr_times)
        starts = np.searchsorted(feature_times, tr_times - TR_SEC)
        pooled = np.zeros((len(tr_times), features.shape[1]), dtype=np.float32)
        for index, (start, stop) in enumerate(zip(starts, edges)):
            if stop <= start:
                continue                       # silence: leave the TR at zero
            block = features[start:stop]
            if self._pooling == 'average':
                pooled[index] = block.mean(axis=0)
            elif self._pooling == 'sum':
                pooled[index] = block.sum(axis=0)
            else:                              # 'last'
                pooled[index] = block[-1]
        return pooled

