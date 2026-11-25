"""THINGS Fig. 2a check: the voxelwise noise ceiling from the packaged trial responses only.

St-Laurent et al. (2026), Scientific Data 13:141, Fig. 2a (p.8), with the method of p.5-6
(``brainscore.benchmarks.noise_ceiling``, the function the THINGS benchmarks use): images
with 3 answered trials (1st, 2nd, 3rd answered trial of an image, in time order). The packaged
responses are a per-voxel shift and scale of the paper's betas, which the method cancels.

    python -m brainscore.data.cneuromod.validation.fig2a_noise_ceiling --packaged <folder> --out <folder>

``<folder>`` is the output of ``package_data('cneuromod-things')``. Writes ``fig2a_noise_ceiling.png``,
``.pdf`` and ``fig2a_noise_ceiling.csv`` (per participant and region: voxels, our median, the
median of the released map on the same voxels) and ``fig2a_voxels.csv`` (per participant: the
voxel-by-voxel agreement with the released map).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brainscore.benchmarks import noise_ceiling
from brainscore.data.cneuromod.data_packaging import things

REGIONS = ('V1', 'V2', 'V3', 'FFA', 'EBA', 'PPA')
# Each region has one mask set in the packaging (``things.MASK_ATLASES``): the participant's own masks
# (pRF, fLoc localizer); sub-06 did no localizer and has only the group parcels of the Kanwisher lab
# warped to its brain, not chosen by any THINGS measure.


def region_rows(table: pd.DataFrame, region: str) -> pd.DataFrame:
    """The voxels of a region (empty: the participant has no mask for it)."""
    return table[table['region'] == region]


def participant_ceiling(folder: Path) -> tuple:
    """``(voxel table with the noise ceiling, images used)`` of one participant's packaged sessions."""
    import xarray as xr
    sessions = []
    for path in sorted(folder.glob('ses-*/trials.nc')):
        session = xr.open_dataarray(path)
        sessions.append(session.isel(neuroid=session['region'].values != 'cortex').load())   # the masks only
    if not sessions:
        raise FileNotFoundError(f'no ses-*/trials.nc under {folder}: package the full participant first')
    neuroid = sessions[0]['neuroid']
    first = ~pd.Index(neuroid['voxel'].values).duplicated()        # a voxel in several masks counts once
    responses = np.concatenate([s.values[:, first] for s in sessions])
    image = np.concatenate([s['stimulus_id'].values for s in sessions])
    response = np.concatenate([s['response'].values for s in sessions])
    ceiling, used = noise_ceiling(responses, image, things.answered_viewing(image, response))
    by_voxel = pd.Series(ceiling, index=neuroid['voxel'].values[first])
    table = pd.DataFrame({name: neuroid[name].values for name in ('atlas', 'region', 'voxel', 'i', 'j', 'k')})
    table['ours'] = by_voxel.loc[table['voxel']].to_numpy()
    return table, used


def released(table: pd.DataFrame, folder: Path) -> np.ndarray:
    """The released noise-ceiling map (copied by the packaging) at the voxels of ``table``."""
    import nibabel
    path, = folder.glob('glmsingle/*_stat-noiseCeilings_statmap.nii.gz')
    values = np.asarray(nibabel.load(path).dataobj)
    return values[table['i'], table['j'], table['k']]


def draw(tables: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(tables), 1, figsize=(7, 2.3 * len(tables)), sharex=True, sharey=True, squeeze=False)
    for ax, (participant, table) in zip(axes[:, 0], tables.items()):
        for x, region in enumerate(REGIONS):
            rows = region_rows(table, region)
            if rows.empty:
                ax.text(x, 1, 'no mask', ha='center', fontsize=7, color='0.5')
                continue
            parts = ax.violinplot(rows['ours'], [x], widths=0.8, showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor(plt.cm.magma(min(rows['ours'].median() / 35, 1) * 0.85))
                body.set_alpha(0.8)
            ax.boxplot(rows['ours'], positions=[x], widths=0.15, showfliers=False)
            ax.plot(x + 0.3, rows['released'].median(), marker='<', color='tab:cyan', ms=6)
        group = 'fLocVisionTask' not in set(table['atlas'])
        ax.set_title(participant + ('   no pRF masks; FFA, EBA, PPA: Kanwisher group parcels' if group else ''),
                     fontsize=9, loc='left')
        ax.set_ylabel('noise ceiling (%)')
        ax.set_ylim(0, None)
    axes[-1, 0].set_xticks(range(len(REGIONS)), REGIONS)
    axes[0, 0].plot([], [], marker='<', ls='', color='tab:cyan', label='released map, median')
    axes[0, 0].legend(fontsize=7, loc='upper right')
    fig.suptitle('Voxelwise noise ceiling (% variance), from the packaged trial responses', fontsize=10)
    fig.tight_layout()
    for suffix in ('.png', '.pdf'):
        fig.savefig(path.with_suffix(suffix), dpi=200)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packaged', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    tables, regions, voxels = {}, [], []
    for folder in sorted(args.packaged.glob('sub-*')):
        table, used = participant_ceiling(folder)
        table['released'] = released(table, folder)
        tables[folder.name] = table
        unique = table.drop_duplicates('voxel')
        voxels.append({'participant': folder.name, 'images': used, 'voxels': len(unique),
                       'r': np.corrcoef(unique['ours'], unique['released'])[0, 1],
                       'largest_abs_difference': np.abs(unique['ours'] - unique['released']).max(),
                       'ours_max': unique['ours'].max(), 'released_max': unique['released'].max()})
        for region in REGIONS:
            rows = region_rows(table, region)
            if len(rows):
                regions.append({'participant': folder.name, 'region': region, 'atlas': rows['atlas'].iloc[0],
                                'voxels': len(rows), 'ours_median': rows['ours'].median(),
                                'released_median': rows['released'].median()})
    pd.DataFrame(regions).to_csv(args.out / 'fig2a_noise_ceiling.csv', index=False, float_format='%.4f')
    pd.DataFrame(voxels).to_csv(args.out / 'fig2a_voxels.csv', index=False, float_format='%.6g')
    draw(tables, args.out / 'fig2a_noise_ceiling.png')
    print(pd.DataFrame(voxels).to_string(index=False))
    print(pd.DataFrame(regions).to_string(index=False))


if __name__ == '__main__':
    main()
