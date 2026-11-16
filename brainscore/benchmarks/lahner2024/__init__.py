from functools import partial

from brainscore import benchmark_registry
from brainscore.data.lahner2024 import MOTION_COLUMNS
from .auditory_roi import build_auditory_mask
from .benchmark import Lahner2024BOLDMoments, VISUAL_ROI_CEILING
from .benchmark_timeresolved import (
    Lahner2024BOLDMoments_timeresolved,
    ParticipantRunBlocks,
    RunBlocks,
)

# One registry line per variant: the class, the full identifier (the registry key) and the parts
# that make the variant.
# Two classes, one per neural target: GLM betas (one estimate per clip) and TR
# time series. The parts:
# - `channels`: the visual stream only (records neural:IT), or the visual stream
#   and the sound track (also records neural:A1);
# - the voxels: all, `reliability_threshold=0.3` (`-visualROI`: split-half
#   reliability at least 0.3, chosen by reliability anywhere on the cortex, not by
#   an atlas). The GLM class follows the protocol of the data set for every variant:
#   one fit on the 933 training-set videos of the assembly, the correlation on the 93
#   test-set videos (id above 1000), and the mask from the reliability on the
#   training-set videos. The time-resolved class measures its mask on the 93 test-set
#   videos of the GLM assembly, scores only the `train` task runs (their videos are
#   the training set) and holds out blocks of 40 runs per participant, which keeps
#   one participant per block; or `voxel_mask_fn=build_auditory_mask`
#   (`-auditoryROI`: early auditory cortex, lateral STG and the planum, Destrieux
#   atlas on fsaverage5);
# - for TR time series, the fold rule and the confounds regressed out of the BOLD.
# Only the two GLM-beta visual-ROI variants divide by a measured ceiling
# (`VISUAL_ROI_CEILING`): per voxel, sqrt of the split-half reliability on the videos
# they score (the 93 test-set videos, each shown 10 times to a participant); the
# other variants return the raw correlation. Every line is version 2 (see the two
# module docstrings for what changed from version 1).
_AUDIOVISUAL = ('vision:image', 'vision:video', 'audio')

# GLM betas: one beta per voxel per stimulus per repetition.
benchmark_registry['Lahner2024-fMRI-naturalistic'] = partial(
    Lahner2024BOLDMoments, identifier='Lahner2024-fMRI-naturalistic')
benchmark_registry['Lahner2024-fMRI-naturalistic-visualROI'] = partial(
    Lahner2024BOLDMoments, reliability_threshold=0.3, identifier='Lahner2024-fMRI-naturalistic-visualROI',
    ceiling=VISUAL_ROI_CEILING)
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal'] = partial(
    Lahner2024BOLDMoments, channels=_AUDIOVISUAL, identifier='Lahner2024-fMRI-naturalistic-multimodal')
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal-visualROI'] = partial(
    Lahner2024BOLDMoments, channels=_AUDIOVISUAL, reliability_threshold=0.3,
    identifier='Lahner2024-fMRI-naturalistic-multimodal-visualROI', ceiling=VISUAL_ROI_CEILING)
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal-auditoryROI'] = partial(
    Lahner2024BOLDMoments, channels=_AUDIOVISUAL, voxel_mask_fn=build_auditory_mask,
    identifier='Lahner2024-fMRI-naturalistic-multimodal-auditoryROI')

# TR time series: clip-mean onset-impulse HRF model, per-voxel ridge, whole runs
# held out (blocks of 52 runs by default, the runs of one participant; blocks of 40
# for the `-visualROI` variants, which score the 40 `train` task runs of a participant);
# events selected by (subject, session, task, run).
# `-improved`: per-run motion regression (9 fmriprep confounds, residuals replace the
# BOLD before the z-score) and blocks of 10 runs held out inside each participant,
# per-voxel r averaged across participants. No stored result measures the effect
# of these changes on the score.
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved'] = partial(
    Lahner2024BOLDMoments_timeresolved, identifier='Lahner2024-fMRI-naturalistic-timeresolved')
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-visualROI'] = partial(
    Lahner2024BOLDMoments_timeresolved, folds=RunBlocks(40), reliability_threshold=0.3,
    identifier='Lahner2024-fMRI-naturalistic-timeresolved-visualROI')
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-improved'] = partial(
    Lahner2024BOLDMoments_timeresolved, folds=ParticipantRunBlocks(10),
    confounds=MOTION_COLUMNS, identifier='Lahner2024-fMRI-naturalistic-timeresolved-improved')
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-improved-visualROI'] = partial(
    Lahner2024BOLDMoments_timeresolved, folds=ParticipantRunBlocks(10),
    confounds=MOTION_COLUMNS, reliability_threshold=0.3,
    identifier='Lahner2024-fMRI-naturalistic-timeresolved-improved-visualROI')
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-multimodal'] = partial(
    Lahner2024BOLDMoments_timeresolved, channels=_AUDIOVISUAL,
    identifier='Lahner2024-fMRI-naturalistic-timeresolved-multimodal')
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-multimodal-visualROI'] = partial(
    Lahner2024BOLDMoments_timeresolved, channels=_AUDIOVISUAL, folds=RunBlocks(40),
    reliability_threshold=0.3,
    identifier='Lahner2024-fMRI-naturalistic-timeresolved-multimodal-visualROI')
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-multimodal-auditoryROI'] = partial(
    Lahner2024BOLDMoments_timeresolved, channels=_AUDIOVISUAL, voxel_mask_fn=build_auditory_mask,
    identifier='Lahner2024-fMRI-naturalistic-timeresolved-multimodal-auditoryROI')
