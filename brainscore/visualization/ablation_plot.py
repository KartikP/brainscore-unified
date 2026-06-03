"""Figures for the ablation / state-change story: a model's response, its
response after a targeted lesion, and the quantified effect on a benchmark.

These render the four panels the showcase needs for the ablation capability:
  1. the model's response to a stimulus (a unit x stimulus activation heatmap),
  2. the response after a :class:`StateChange` ablation,
  3. the difference, and
  4. a bar plot with error bars comparing baseline / lesioned / control /
     restored benchmark scores — the quantitative payoff.

matplotlib only; no model or scoring dependency, so they are testable offline
and reusable in any notebook or the website pipeline.
"""
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np


# Stable colours per condition so every ablation figure across the site reads
# the same way.
_CONDITION_COLORS = {
    'baseline': '#3b7dd8',
    'lesioned': '#d8483b',
    'random': '#9aa0a6',
    'random control': '#9aa0a6',
    'opposite': '#c98a1b',
    'restored': '#3bb273',
}


def _color_for(name: str) -> str:
    return _CONDITION_COLORS.get(name.lower(), '#6c6c6c')


def ablation_effect_bar(conditions: Dict[str, Union[Sequence[float], Tuple[float, float]]],
                        *, ylabel: str = 'benchmark score', title: Optional[str] = None,
                        chance: Optional[float] = None, out_png: Optional[str] = None):
    """Bar plot with error bars comparing benchmark scores across conditions.

    ``conditions`` maps a condition name (``baseline``, ``lesioned``,
    ``random``, ``restored``, ...) to either a sequence of per-seed/per-fold
    scores (mean + SEM computed) or a ``(mean, sem)`` tuple. Draws an optional
    dashed ``chance`` reference line — the null floor the effect must respect.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names, means, sems = [], [], []
    for name, v in conditions.items():
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 1 and arr.size == 2:
            mean, sem = float(arr[0]), float(arr[1])
        else:
            mean = float(np.nanmean(arr))
            sem = float(np.nanstd(arr) / np.sqrt(max(1, np.isfinite(arr).sum())))
        names.append(name); means.append(mean); sems.append(sem)

    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 1.5, 3.4))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=sems, capsize=4,
           color=[_color_for(n) for n in names], edgecolor='black', linewidth=0.5)
    if chance is not None:
        ax.axhline(chance, ls='--', color='black', lw=1, alpha=0.6)
        ax.text(len(names) - 0.5, chance, ' chance', va='bottom', ha='right',
                fontsize=8, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig


def response_heatmap(activations: np.ndarray, *, title: Optional[str] = None,
                     cmap: str = 'viridis', vmin: Optional[float] = None,
                     vmax: Optional[float] = None, xlabel: str = 'unit',
                     ylabel: str = 'stimulus', out_png: Optional[str] = None):
    """Render a ``(stimulus, unit)`` activation matrix as a heatmap.

    Use one for the intact model and one for the lesioned model to show, at a
    glance, which units went dark under ablation.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    activations = np.asarray(activations, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    im = ax.imshow(activations, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig


def before_after_difference(intact: np.ndarray, lesioned: np.ndarray, *,
                            title: Optional[str] = None,
                            out_png: Optional[str] = None):
    """Three-panel intact | lesioned | (intact - lesioned) heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    intact = np.asarray(intact, dtype=float)
    lesioned = np.asarray(lesioned, dtype=float)
    diff = intact - lesioned
    vmax = float(np.nanmax(np.abs(np.concatenate([intact.ravel(), lesioned.ravel()]))))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, data, name, cmap in [
        (axes[0], intact, 'intact', 'viridis'),
        (axes[1], lesioned, 'lesioned', 'viridis'),
        (axes[2], diff, 'intact − lesioned', 'RdBu_r')]:
        vm = vmax if name != 'intact − lesioned' else float(np.nanmax(np.abs(diff)) or 1)
        kw = dict(vmin=0, vmax=vmax) if name != 'intact − lesioned' else dict(vmin=-vm, vmax=vm)
        im = ax.imshow(data, aspect='auto', cmap=cmap, **kw)
        ax.set_title(name, fontsize=10); ax.set_xlabel('unit'); ax.set_ylabel('stimulus')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig
