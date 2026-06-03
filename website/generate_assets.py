"""Render the cortical-surface figures the showcase site uses, one per input
type, emphasizing the Schaefer 7-networks that input modality is known to drive.

These are illustrative maps (network-structured, not a single measured run) so
the site can show "where each input's predicted responses concentrate" without
claiming a specific per-parcel r. Run on EC2 (nilearn + fsaverage assets), CPU
only. Outputs PNGs into ./assets/.
"""
import os
import numpy as np

from brainscore.visualization import (cortical_surface_map, quickbrain_outline_map,
                                       fetch_schaefer_fsaverage_annot)
from brainscore.visualization.brain_map import SCHAEFER_7NETWORKS

# Render backend: 'quickbrain' (stylized brain outline) or 'nilearn' (inflated surface).
BACKEND = os.environ.get('BRAIN_BACKEND', 'quickbrain')


def render(vals, out_png, title=None):
    if BACKEND == 'quickbrain':
        return quickbrain_outline_map(vals, hemi='left', view='lateral',
                                      threshold=None, colorbar=True,
                                      title=title, out_png=out_png)
    return cortical_surface_map(vals, hemi='left', view='lateral',
                                vmin=0.0, vmax=0.5, title=title, out_png=out_png)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, 'assets')
os.makedirs(ASSETS, exist_ok=True)

# network emphasis per input type (network -> weight); others get a small base.
# Slugs must match app.js slug(type): "video + audio + text" -> "videoaudiotext".
EMPHASIS = {
    'image':           {'Vis': 1.0, 'DorsAttn': 0.4},
    'text':            {'Default': 1.0, 'Cont': 0.7, 'SalVentAttn': 0.3},
    'audio':           {'SomMot': 1.0, 'DorsAttn': 0.3},  # auditory ~ SomMot/temporal
    'video':           {'Vis': 0.9, 'DorsAttn': 0.7, 'SomMot': 0.5},
    'videoaudio':      {'Vis': 0.7, 'SomMot': 0.9, 'DorsAttn': 0.5, 'Default': 0.4},
    'videoaudiotext':  {'Vis': 0.75, 'SomMot': 0.75, 'Default': 0.85, 'Cont': 0.6,
                        'DorsAttn': 0.5, 'SalVentAttn': 0.4},  # broad, whole-cortex
}


def network_of(names):
    idx = np.full(len(names), -1, dtype=int)
    for i, nm in enumerate(names):
        for n, net in enumerate(SCHAEFER_7NETWORKS):
            if net.lower() in str(nm).lower():
                idx[i] = n
                break
    return idx


def main():
    lh_lab, rh_lab, lh_names, rh_names = fetch_schaefer_fsaverage_annot(
        n_parcels=1000, networks=7, resolution='fsaverage5')
    # annot names are per-label (1-indexed, 0=medial wall); parcel j -> name j+1.
    lh_parcel_names = lh_names[1:501]
    rh_parcel_names = rh_names[1:501]
    names = list(lh_parcel_names) + list(rh_parcel_names)
    net = network_of(names)
    rng = np.random.RandomState(0)

    for key, emph in EMPHASIS.items():
        vals = 0.06 + 0.04 * rng.rand(1000)            # low base everywhere
        for n, net_name in enumerate(SCHAEFER_7NETWORKS):
            w = emph.get(net_name, 0.0)
            if w:
                mask = net == n
                vals[mask] = (0.18 + 0.30 * w) + 0.05 * rng.rand(mask.sum())
        vals = np.clip(vals, 0, 0.6)
        out = os.path.join(ASSETS, f'cortex_{key}.png')
        render(vals, out)
        print('wrote', out, 'mean', round(float(vals.mean()), 3))

    # also a generic demo (kept as the fallback)
    vals = np.clip(0.05 + 0.4 * rng.rand(1000), 0, 0.6)
    render(vals, os.path.join(ASSETS, 'cortex_demo.png'))
    print(f'done (backend={BACKEND})')


if __name__ == '__main__':
    main()
