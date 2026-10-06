"""Render notebook 16's saved measurements without rerunning GPT-2.

The NPZ contains measured held-out correlations, not simulated brain activity.
Surface coordinates follow the LeBel assembly's fsaverage5, left-first order.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from nilearn import datasets, plotting

ASSETS = Path(__file__).resolve().parents[1] / 'assets' / 'whole_brain'
with np.load(ASSETS / 'gpt2_lebel_scores.npz', allow_pickle=False) as values:
    scores = values['per_vertex']
    indices = values['target_index']
metadata = json.loads((ASSETS / 'measurements.json').read_text())
assert metadata['measured'] is True
assert len(scores) == len(indices) == 20484
assert np.array_equal(np.sort(indices), np.arange(20484))
assert np.isclose(np.nanmedian(scores), metadata['median_r'])
surface_scores = np.full(20484, np.nan)
surface_scores[indices] = scores
finite = scores[np.isfinite(scores)]
limit = float(np.ceil(np.abs(finite).max() * 10) / 10)
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
})

# All four views share a symmetric scale; negative values remain visible.
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
fig = plt.figure(figsize=(10, 7))
for number, (hemi, view) in enumerate([
    ('left', 'lateral'), ('right', 'lateral'),
    ('left', 'medial'), ('right', 'medial'),
]):
    ax = fig.add_subplot(2, 2, number + 1, projection='3d')
    half = surface_scores[:10242] if hemi == 'left' else surface_scores[10242:]
    plotting.plot_surf_stat_map(
        fsaverage[f'infl_{hemi}'],
        half,
        hemi=hemi,
        view=view,
        bg_map=fsaverage[f'sulc_{hemi}'],
        cmap='RdBu_r',
        threshold=None,
        vmin=-limit,
        vmax=limit,
        colorbar=False,
        axes=ax,
        figure=fig,
    )
    ax.set_box_aspect(None, zoom=1.3)
    side = 'outside' if view == 'lateral' else 'inside'
    ax.set_title(f'{hemi.capitalize()} hemisphere · {side}', pad=-3)
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.16, hspace=0.03)
fig.suptitle('Where GPT-2 predicts story-listening brain responses', fontweight='bold')
color_axis = fig.add_axes([0.27, 0.12, 0.46, 0.025])
bar = fig.colorbar(
    plt.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap='RdBu_r'),
    cax=color_axis,
    orientation='horizontal',
)
bar.set_label('Held-out correlation (Pearson r)')
fig.text(
    0.5, 0.022,
    'Blue: opposite pattern     White: near zero     Red: matching pattern',
    ha='center', fontsize=10,
)
fig.savefig(ASSETS / 'cortical_predictions.png', dpi=180, facecolor='white')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(finite, bins=60, color='#0072B2', edgecolor='white', linewidth=0.3)
ax.axvline(0, color='0.4', linestyle='--', label='Zero correlation')
ax.axvline(
    metadata['median_r'], color='#D55E00', linewidth=2,
    label=f"Median = {metadata['median_r']:.3f}",
)
ax.set(
    xlabel='Held-out correlation (Pearson r)', ylabel='Cortical locations',
    title=f'Prediction accuracy across {len(finite):,} cortical locations',
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(ASSETS / 'score_distribution.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print('Saved whole-brain figures:', ASSETS)
