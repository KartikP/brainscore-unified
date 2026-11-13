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

# One registry line per variant: the class and the parts that make the variant.
# Two classes, one per neural target: GLM betas (one estimate per clip) and TR
# time series. The parts:
# - `channels`: the visual stream only (records neural:IT), or the visual stream
#   and the sound track (also records neural:A1);
# - the voxels: all, `reliability_threshold=0.3` (`-visualROI`: split-half
#   reliability at least 0.3, about 2000 to 5000 voxels in visual cortex), or
#   `voxel_mask_fn=build_auditory_mask` (`-auditoryROI`: early auditory cortex and
#   the planum, Destrieux atlas on fsaverage5);
# - for TR time series, the fold rule and the confounds regressed out of the BOLD.
# Only the GLM-beta visual ROI divides by a measured ceiling (`VISUAL_ROI_CEILING`);
# the other variants return the raw correlation.
_AUDIOVISUAL = ('vision:image', 'vision:video', 'audio')

# GLM betas: one beta per voxel per stimulus per repetition.
benchmark_registry['Lahner2024-fMRI-naturalistic'] = Lahner2024BOLDMoments
benchmark_registry['Lahner2024-fMRI-naturalistic-visualROI'] = partial(
    Lahner2024BOLDMoments, reliability_threshold=0.3, identifier_suffix='-visualROI',
    ceiling=VISUAL_ROI_CEILING)
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal'] = partial(
    Lahner2024BOLDMoments, channels=_AUDIOVISUAL, identifier_suffix='-multimodal')
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal-visualROI'] = partial(
    Lahner2024BOLDMoments, channels=_AUDIOVISUAL, reliability_threshold=0.3,
    identifier_suffix='-multimodal-visualROI')
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal-auditoryROI'] = partial(
    Lahner2024BOLDMoments, channels=_AUDIOVISUAL, voxel_mask_fn=build_auditory_mask,
    identifier_suffix='-multimodal-auditoryROI')

# TR time series: clip-mean onset-impulse HRF model, per-voxel ridge, whole runs
# held out (blocks of 52 runs by default). Version 2 (default) selects events by
# (subject, session, task, run); version 1 (defect F6, task omitted) stays
# reachable for reproducing historical scores: load_benchmark(identifier, version=1).
# `-improved`: per-run motion regression (9 fmriprep confounds, residuals replace the
# BOLD before the z-score) and blocks of 10 runs held out inside each participant,
# per-voxel r averaged across participants. The ROI raw r moved by +0.001 to +0.016
# and the whole-cortex r fell by about half.
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved'] = (
    Lahner2024BOLDMoments_timeresolved)
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-visualROI'] = partial(
    Lahner2024BOLDMoments_timeresolved, reliability_threshold=0.3,
    identifier_suffix='-timeresolved-visualROI')
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-improved'] = partial(
    Lahner2024BOLDMoments_timeresolved, folds=ParticipantRunBlocks(10),
    confounds=MOTION_COLUMNS, identifier_suffix='-timeresolved-improved')
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-improved-visualROI'] = partial(
    Lahner2024BOLDMoments_timeresolved, folds=ParticipantRunBlocks(10),
    confounds=MOTION_COLUMNS, reliability_threshold=0.3,
    identifier_suffix='-timeresolved-improved-visualROI')
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-multimodal'] = partial(
    Lahner2024BOLDMoments_timeresolved, channels=_AUDIOVISUAL,
    identifier_suffix='-timeresolved-multimodal')
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-multimodal-visualROI'] = partial(
    Lahner2024BOLDMoments_timeresolved, channels=_AUDIOVISUAL, reliability_threshold=0.3,
    identifier_suffix='-timeresolved-multimodal-visualROI')
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-multimodal-auditoryROI'] = partial(
    Lahner2024BOLDMoments_timeresolved, channels=_AUDIOVISUAL, voxel_mask_fn=build_auditory_mask,
    identifier_suffix='-timeresolved-multimodal-auditoryROI')
