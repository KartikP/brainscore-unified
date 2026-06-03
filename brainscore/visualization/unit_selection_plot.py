"""Figures for the unit-selection / composite-selection capability: *which*
population of model units, across *which* layers, a benchmark reads out from.

A :class:`CompositeSelector` gathers units from several layers into one "region".
These figures make that population legible:
  * :func:`composite_selection_map` — a layers x units mask showing exactly which
    units are selected in each layer.
  * :func:`units_per_layer_bar` — how the selected population is distributed
    across depth.
  * :func:`selectivity_histogram` — the score (e.g. Cohen's d) that drove the
    selection, with the selected units highlighted against the full population.

matplotlib only; no model dependency.
"""
from typing import Dict, Optional, Sequence

import numpy as np


def composite_selection_map(selection_by_layer: Dict[str, Sequence[int]],
                            units_per_layer: Dict[str, int], *,
                            max_cols: int = 400, title: Optional[str] = None,
                            out_png: Optional[str] = None):
    """Render a ``layers x units`` binary mask of the selected population.

    ``selection_by_layer`` maps a layer path to the indices of its selected
    units; ``units_per_layer`` gives each layer's total unit count. Each layer
    is a row; selected units are bright. Wide layers are column-downsampled to
    ``max_cols`` (a column is "on" if any unit it covers is selected) so very
    large layers stay legible.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers = list(selection_by_layer.keys())
    cols = min(max_cols, max(units_per_layer[l] for l in layers))
    mask = np.zeros((len(layers), cols), dtype=float)
    for r, layer in enumerate(layers):
        total = units_per_layer[layer]
        sel = np.asarray(list(selection_by_layer[layer]), dtype=int)
        if sel.size:
            # map each selected unit to its downsampled column
            colidx = np.clip((sel.astype(float) / max(1, total) * cols).astype(int),
                             0, cols - 1)
            mask[r, colidx] = 1.0

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(layers) + 1.2))
    ax.imshow(mask, aspect='auto', cmap='magma', interpolation='nearest')
    ax.set_yticks(np.arange(len(layers))); ax.set_yticklabels(layers, fontsize=8)
    ax.set_xlabel(f'unit index (downsampled to {cols} columns)', fontsize=9)
    n_sel = sum(len(list(v)) for v in selection_by_layer.values())
    ax.set_title(title or f'Composite selection — {n_sel} units across '
                 f'{len(layers)} layers', fontsize=11)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig


def units_per_layer_bar(selection_by_layer: Dict[str, Sequence[int]], *,
                        units_per_layer: Optional[Dict[str, int]] = None,
                        as_fraction: bool = False, title: Optional[str] = None,
                        out_png: Optional[str] = None):
    """Bar plot of how many selected units come from each layer (the depth
    profile of the readout population). With ``as_fraction`` and
    ``units_per_layer``, shows the fraction of each layer that was selected.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers = list(selection_by_layer.keys())
    counts = [len(list(selection_by_layer[l])) for l in layers]
    if as_fraction and units_per_layer:
        vals = [c / max(1, units_per_layer[l]) for c, l in zip(counts, layers)]
        ylabel = 'fraction of layer selected'
    else:
        vals = counts
        ylabel = 'selected units'
    fig, ax = plt.subplots(figsize=(1.0 * len(layers) + 1.5, 3.2))
    x = np.arange(len(layers))
    ax.bar(x, vals, color='#6a3d9a', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(layers, rotation=30, ha='right', fontsize=8)
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


def selectivity_histogram(scores: np.ndarray, selected_mask: np.ndarray, *,
                          score_name: str = "Cohen's d", title: Optional[str] = None,
                          out_png: Optional[str] = None):
    """Histogram of a per-unit selectivity score with the selected units
    overlaid — shows the threshold logic behind the selection.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scores = np.asarray(scores, dtype=float)
    selected_mask = np.asarray(selected_mask, dtype=bool)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(scores[~selected_mask], bins=40, color='#cccccc', label='not selected')
    ax.hist(scores[selected_mask], bins=40, color='#d8483b', label='selected')
    ax.set_xlabel(score_name, fontsize=10); ax.set_ylabel('units', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    if title:
        ax.set_title(title, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig
