"""Layer-contribution-per-modality heatmap — the MIRAGE Figure 4 style figure.

A modality x layer heatmap of how much each layer of a model contributes to a
modality-specific readout / brain prediction. MIRAGE renders cross-attention
weights of Qwen3-Omni's per-modality modules over its 48 language layers; the
analogue here is the per-layer brain-prediction score (or attention/contribution
weight) for each modality tower, normalized per modality.

matplotlib only.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np


def layer_modality_heatmap(contribution: Dict[str, Sequence[float]], *,
                           layer_axis: Optional[Sequence[int]] = None,
                           normalize: str = 'row', cmap: str = 'magma',
                           cbar_label: str = 'contribution',
                           title: Optional[str] = None,
                           out_png: Optional[str] = None):
    """Render a ``modality x layer`` heatmap.

    :param contribution: ``{modality_name: per-layer values}``. Rows need not be
        the same length; shorter rows are right-padded with NaN so towers with
        different depths share one axis (NaNs render transparent).
    :param layer_axis: optional explicit layer indices for the x ticks.
    :param normalize: ``'row'`` scales each modality to its own max (matches
        MIRAGE's per-modality readout weighting); ``'none'`` leaves raw values;
        ``'global'`` scales all rows by the global max.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    modalities = list(contribution.keys())
    max_len = max(len(v) for v in contribution.values())
    grid = np.full((len(modalities), max_len), np.nan)
    for i, m in enumerate(modalities):
        v = np.asarray(contribution[m], dtype=float)
        grid[i, :len(v)] = v

    if normalize == 'row':
        for i in range(grid.shape[0]):
            row = grid[i]
            mx = np.nanmax(row) if np.isfinite(row).any() else 1.0
            if mx > 0:
                grid[i] = row / mx
    elif normalize == 'global':
        mx = np.nanmax(grid)
        if mx > 0:
            grid = grid / mx

    fig, ax = plt.subplots(figsize=(max(6, max_len * 0.28), 0.7 * len(modalities) + 1.6))
    im = ax.imshow(grid, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_yticks(range(len(modalities)))
    ax.set_yticklabels(modalities, fontsize=11)
    ax.set_xlabel('layer', fontsize=11)
    ax.set_ylabel('modality', fontsize=11)
    if layer_axis is not None:
        step = max(1, max_len // 10)
        ax.set_xticks(range(0, max_len, step))
        ax.set_xticklabels([layer_axis[i] for i in range(0, max_len, step)], fontsize=9)
    if title:
        ax.set_title(title, fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(cbar_label, fontsize=9)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig
