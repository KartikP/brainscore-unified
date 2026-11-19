from functools import partial

from brainscore import benchmark_registry
from .auditory_roi import build_auditory_mask


def _load(**configuration):
    from .benchmark import Lahner2024BOLDMoments
    return Lahner2024BOLDMoments(**configuration)


# Noise ceiling for the visual-ROI voxel set. For a predictivity correlation the ceiling
# is sqrt(reliability): the highest r a perfect model could reach given measurement
# noise. Under the protocol of the data set the constant is the median of
# sqrt(reliability on the scored videos, the 93 test-set videos) over the voxels of the
# mask (reliability of at least 0.3 on the 933 training-set videos) that have a
# positive ceiling. Measured on the hosted data: the mask has 3958 voxels, every one
# with a positive ceiling, and this is their median. It is what ``benchmark.ceiling``
# reports for both ``-visualROI`` variants; the score divides each voxel by its own
# ceiling and does not read this value.
VISUAL_ROI_CEILING = 0.8515666971964203

# One registry line per variant: the class, the full identifier (the registry key) and the parts
# that make the variant. The registered class fits GLM betas (one estimate per clip).
# The parts:
# - `channels`: the visual stream only (records neural:IT), or the visual stream
#   and the sound track (also records neural:A1);
# - the voxels: all, `reliability_threshold=0.3` (`-visualROI`: split-half
#   reliability at least 0.3, chosen by reliability anywhere on the cortex, not by
#   an atlas), or `voxel_mask_fn=build_auditory_mask` (`-auditoryROI`: early
#   auditory cortex, lateral STG and the planum, Destrieux atlas on fsaverage5).
# The class follows the protocol of the data set for every variant: one fit on the
# 933 training-set videos of the assembly, the correlation on the 93 test-set videos
# (id above 1000), and the mask from the reliability on the training-set videos.
# Only the two visual-ROI variants divide by a measured ceiling (`VISUAL_ROI_CEILING`):
# per voxel, sqrt of the split-half reliability on the videos they score (the 93
# test-set videos, each shown 10 times to a participant); the other variants return
# the raw correlation. Every line is version 2 (see the module docstring of
# `benchmark.py` for what changed from version 1).
_AUDIOVISUAL = ('vision:image', 'vision:video', 'audio')

benchmark_registry['Lahner2024-fMRI-naturalistic'] = partial(
    _load, identifier='Lahner2024-fMRI-naturalistic')
benchmark_registry['Lahner2024-fMRI-naturalistic-visualROI'] = partial(
    _load, reliability_threshold=0.3, identifier='Lahner2024-fMRI-naturalistic-visualROI',
    ceiling=VISUAL_ROI_CEILING)
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal'] = partial(
    _load, channels=_AUDIOVISUAL, identifier='Lahner2024-fMRI-naturalistic-multimodal')
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal-visualROI'] = partial(
    _load, channels=_AUDIOVISUAL, reliability_threshold=0.3,
    identifier='Lahner2024-fMRI-naturalistic-multimodal-visualROI', ceiling=VISUAL_ROI_CEILING)
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal-auditoryROI'] = partial(
    _load, channels=_AUDIOVISUAL, voxel_mask_fn=build_auditory_mask,
    identifier='Lahner2024-fMRI-naturalistic-multimodal-auditoryROI')

# The TR time-series class (`benchmark_timeresolved.Lahner2024BOLDMoments_timeresolved`) is not registered.
# It does not separate a trained model from the same model with random weights: of
# the seven variants that were registered, none scored more than 0.016 above its
# random-weight floor and two variants scored below the floor (measured 2026-11-16 on
# the hosted data with this code, models clip-vit-b-32, videomae-base, vjepa1-vitl,
# clip-wav2vec2 and vjepa1-wav2vec2 and their random-weight floors).
# To use it, build it directly, for example
# `Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(40), reliability_threshold=0.3,
# identifier='...')`; its docstring lists the parts. A time-resolved variant that
# follows the data set's own analysis is planned.
