"""Scaling-curve figures: capability score as a function of model quality,
from a bad model (random init / chance) up to a good one.

This is the headline validation of the whole interface. A benchmark that is
worth running should *track model quality*: its score should climb as the model
improves and sit near the matched null floor for a random-init model. A flat
curve means the benchmark is saturated, noise-limited, or measuring something
orthogonal to the quality axis. Plotting every capability this way — neural
encoding, behavior, ablation magnitude, embodied success — on one model ladder
is how we show, at a glance, that the unified interface measures something real.

matplotlib only.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np


def scaling_curve_single(models: Sequence[str], scores: Sequence[float], *,
                         null_floor: Optional[float] = None,
                         capability: str = 'score',
                         ylabel: Optional[str] = None,
                         title: Optional[str] = None,
                         out_png: Optional[str] = None):
    """One capability's scaling curve. ``models`` are ordered bad->good and
    plotted left to right; ``null_floor`` (if given) is a dashed reference.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(1.1 * len(models) + 1.5, 3.6))
    ax.plot(x, scores, '-o', color='#1f77b4', lw=2, ms=7)
    if null_floor is not None:
        ax.axhline(null_floor, ls='--', color='#d8483b', lw=1.5)
        ax.text(len(models) - 1, null_floor, ' null floor', va='bottom',
                ha='right', color='#d8483b', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel(ylabel or capability, fontsize=10)
    ax.set_xlabel('model (worse → better)', fontsize=10)
    ax.set_title(title or f'{capability} scales with model quality', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig


def scaling_curves_grid(models: Sequence[str],
                        scores_by_capability: Dict[str, Sequence[float]], *,
                        null_floors: Optional[Dict[str, float]] = None,
                        ncols: int = 3, title: Optional[str] = None,
                        out_png: Optional[str] = None):
    """A small-multiples grid: one scaling curve per capability, shared model
    ladder. ``scores_by_capability`` maps capability name -> per-model scores
    (same order as ``models``; use ``nan`` where a model doesn't apply).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    caps = list(scores_by_capability.keys())
    null_floors = null_floors or {}
    n = len(caps)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    x = np.arange(len(models))
    for i, cap in enumerate(caps):
        ax = axes[i // ncols][i % ncols]
        scores = np.asarray(scores_by_capability[cap], dtype=float)
        ax.plot(x, scores, '-o', color='#1f77b4', lw=2, ms=6)
        if cap in null_floors:
            ax.axhline(null_floors[cap], ls='--', color='#d8483b', lw=1.3)
        ax.set_title(cap, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha='right', fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
    # blank any unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97] if title else None)
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig


def normalized_scaling_overlay(models: Sequence[str],
                               scores_by_capability: Dict[str, Sequence[float]], *,
                               title: Optional[str] = None,
                               out_png: Optional[str] = None):
    """Overlay every capability on one axis after min-max normalizing each to
    ``[0, 1]`` — shows whether capabilities agree on the quality ordering.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(1.2 * len(models) + 2, 4))
    cmap = plt.get_cmap('tab10')
    for k, (cap, scores) in enumerate(scores_by_capability.items()):
        s = np.asarray(scores, dtype=float)
        finite = s[np.isfinite(s)]
        lo, hi = (finite.min(), finite.max()) if finite.size else (0, 1)
        norm = (s - lo) / (hi - lo) if hi > lo else s * 0
        ax.plot(x, norm, '-o', color=cmap(k % 10), lw=1.8, ms=5, label=cap)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('normalized capability score', fontsize=10)
    ax.set_xlabel('model (worse → better)', fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    if title:
        ax.set_title(title, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig
