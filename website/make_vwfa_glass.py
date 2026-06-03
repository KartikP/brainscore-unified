"""Glass-brain of where the ablated VWF-selective population maps onto the brain.

The functional localizer selects visual-word-form (VWF) units. Honarmand et al.
(Fig 5) show those units align with the human Visual Word Form Area (VWFA) — left
ventral occipitotemporal cortex, MNI ~ [-44, -58, -15]. So "what area is dropped"
when we ablate the population is the VWFA. We render that focus as a quickbrain
glass brain (a Gaussian blob at the VWFA coordinate projected to the cortex).

Run on EC2 (quickbrain + nilearn). CPU only. Outputs assets/vwfa_glass.png.
"""
import os

import numpy as np
import nibabel as nib
import matplotlib as mpl
mpl.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white',
                     'xtick.color': 'white', 'ytick.color': 'white'})
import matplotlib.pyplot as plt
from nilearn import datasets
from scipy.ndimage import gaussian_filter

import quickbrain

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, 'assets')

VWFA_MNI = np.array([-44.0, -58.0, -15.0, 1.0])   # left VWFA (McCandliss/Cohen)


def main():
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
    img = nib.load(atlas['maps']) if isinstance(atlas['maps'], str) else atlas['maps']
    affine, shape = img.affine, img.shape

    vox = np.linalg.inv(affine) @ VWFA_MNI
    vi, vj, vk = (int(round(vox[0])), int(round(vox[1])), int(round(vox[2])))
    vol = np.zeros(shape, dtype=float)
    if 0 <= vi < shape[0] and 0 <= vj < shape[1] and 0 <= vk < shape[2]:
        vol[vi, vj, vk] = 1.0
    vol = gaussian_filter(vol, sigma=6.0)
    if vol.max() > 0:
        vol = vol / vol.max()
    blob = nib.Nifti1Image(vol, affine)

    out = os.path.join(ASSETS, 'vwfa_glass.png')
    # sequential inferno (data is 0+), transparent canvas + light text for the
    # dark site. quickbrain only supports lateral/medial views.
    fig = quickbrain.plot_brain(blob, hemi='left', view='lateral', threshold=0.08,
                                colorbar=True, cmap='inferno', background='transparent',
                                title='ablated VWF population → VWFA')
    f = fig if hasattr(fig, 'savefig') else plt.gcf()
    try:
        f.suptitle('ablated VWF population → VWFA', color='white')
    except Exception:
        pass
    f.savefig(out, dpi=150, bbox_inches='tight', transparent=True)
    plt.close(f)
    print('wrote', out)


if __name__ == '__main__':
    main()
