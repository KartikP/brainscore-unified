# fLoc localizer stimuli

Registered stimulus set: **fLoc** (`load_stimulus_set('fLoc')`).

Source: the functional localizer of the Stanford Vision and Perception Neuroscience Lab,
[VPNL/fLoc](https://github.com/VPNL/fLoc), commit `de6a26cc269a2c7075461a4c839bfd628f225c95`.
The loader clones it at the first use into `<data home>/sources/floc` (about 190 MB).
Brain-Score hosts no copy.

Content: five conditions, two subcategories each, 144 images per subcategory (1,440 images):
faces (adult, child), bodies (body, limb), characters (word, number), objects (car,
instrument), places (house, corridor). Each item is on a phase-scrambled background, and
the luminance and grayscale histograms are matched across images (authors' README). The
repository's `scrambled` folder (the oddball task's image) is not a condition and is not read.

Terms: the repository has no licence file. Its README asks users to acknowledge the
localizer or the stimulus set, for example:

> "We used stimuli included in the fLoc functional localizer package (Stigliani et al., 2015)..."

Citation: Stigliani, A., Weiner, K. S., & Grill-Spector, K. (2015). Temporal processing
capacity in high-level visual cortex is domain specific. *Journal of Neuroscience*, 35(36),
12412-12424.

## Rebuild

### floc

- Source: fLoc (Stigliani et al. 2015), https://github.com/VPNL/fLoc at a pinned commit. The repository holds no licence file; its README asks for an acknowledgement and the citation.
- Access: public, no account.
- Needs: git; about 190 MB.
- Command: none: nothing to package.
- Writes: Nothing to package: the first `load_stimulus_set("fLoc")` clones the repository into `~/brainscore-data/sources/floc` and reads its images in place.
- Check: nothing to compare.
