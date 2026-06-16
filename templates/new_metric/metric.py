"""TODO: one-line description of what your metric measures and WHICH alignment
axis it is (predictivity / representational / topographic / behavioral / causal).

A metric compares two assemblies and returns a Score in roughly [0, 1] (1 = identical,
0 = no match). It must NOT depend on a particular model — only on the two assemblies.
"""
import numpy as np

from brainscore_core.metrics import Metric, Score


class YourMetric(Metric):
    """TODO: describe the comparison. Document what coords each assembly must carry
    (e.g. matching `presentation` for RSA; per-unit `tissue_x`/`tissue_y` for topographic)."""

    def __init__(self, **params):
        # TODO: stash any configuration (n_bins, n_components, alpha, ...).
        self.params = params

    def __call__(self, assembly1, assembly2) -> Score:
        # assembly1 is typically the model prediction, assembly2 the target measurement.
        # Convention: (presentation, neuroid). Use .transpose('presentation', 'neuroid')
        # if you need a fixed axis order; read neuroid coords via walk_coords(assembly).
        a = np.asarray(assembly1.transpose('presentation', 'neuroid').values)
        b = np.asarray(assembly2.transpose('presentation', 'neuroid').values)

        # TODO: replace with your comparison. Fail loudly on a shape/coord mismatch
        # rather than returning a misleading number.
        value = float(...)  # your alignment scalar

        score = Score(value)
        score.attrs['metric'] = 'your-metric'
        # Optional: keep per-unit / per-fold detail for plotting + debugging.
        # score.attrs['raw'] = ...
        return score
