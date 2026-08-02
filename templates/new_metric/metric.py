"""TODO: one-line description of what your metric measures and WHICH alignment
axis it is (predictivity / representational / topographic / behavioral / causal).

A metric compares two assemblies and returns a Score in roughly [0, 1] (1 = identical,
0 = no match). It must NOT depend on a particular model — only on the two assemblies.

**This file runs as-is.** It implements mean per-unit Pearson correlation — a real, if
plain, predictivity metric that scores identical inputs at 1.0 and unrelated inputs near
0. Copy the folder, run `pytest`, watch it pass, then replace the marked block with your
own comparison and keep the tests green as you go.
"""
import numpy as np

from brainscore_core.metrics import Metric, Score


class YourMetric(Metric):
    """Mean per-unit Pearson correlation between two (presentation, neuroid) assemblies.

    TODO: describe YOUR comparison here, and document what coords each assembly must
    carry (e.g. matching `presentation` for RSA; per-unit `tissue_x`/`tissue_y` for
    topographic).
    """

    def __init__(self, **params):
        # TODO: stash any configuration (n_bins, n_components, alpha, ...).
        self.params = params

    def __call__(self, assembly1, assembly2) -> Score:
        # assembly1 is typically the model prediction, assembly2 the target measurement.
        # Convention: (presentation, neuroid). Use .transpose('presentation', 'neuroid')
        # for a fixed axis order; read neuroid coords via walk_coords(assembly).
        a = np.asarray(assembly1.transpose('presentation', 'neuroid').values, dtype=float)
        b = np.asarray(assembly2.transpose('presentation', 'neuroid').values, dtype=float)

        # Fail loudly on a mismatch rather than returning a misleading number.
        # Worth keeping whatever your metric ends up doing.
        if a.shape != b.shape:
            raise ValueError('assemblies must have matching shape to compare per unit; '
                             f'got {a.shape} and {b.shape}')
        if a.shape[0] < 2:
            raise ValueError('need at least 2 presentations to correlate')

        # ---- BEGIN: replace this block with your own comparison ----------------
        # Correlate each unit's profile across presentations, then average.
        a_c = a - a.mean(axis=0, keepdims=True)
        b_c = b - b.mean(axis=0, keepdims=True)
        denom = np.linalg.norm(a_c, axis=0) * np.linalg.norm(b_c, axis=0)
        # A unit with no variance carries no information; drop it rather than emit NaN.
        usable = denom > 0
        if not usable.any():
            raise ValueError('no units with non-zero variance in both assemblies')
        per_unit = (a_c[:, usable] * b_c[:, usable]).sum(axis=0) / denom[usable]
        value = float(np.mean(per_unit))
        # ---- END ---------------------------------------------------------------

        score = Score(value)
        score.attrs['metric'] = 'your-metric'
        # Keeping per-unit detail costs nothing and makes later debugging and
        # plotting possible. Most useful metrics expose something like this.
        score.attrs['raw'] = per_unit
        score.attrs['n_units_scored'] = int(usable.sum())
        return score
