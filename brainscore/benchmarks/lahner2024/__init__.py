from functools import partial

from brainscore import benchmark_registry
from .auditory_roi import build_auditory_mask
from .benchmark import Lahner2024BOLDMoments, VISUAL_ROI_CEILING
from .benchmark_timeresolved import Lahner2024BOLDMoments_timeresolved
from .benchmark_multimodal import Lahner2024BOLDMoments_multimodal
from .benchmark_timeresolved_multimodal import Lahner2024BOLDMoments_timeresolved_multimodal

# One registry line per variant: the class and the arguments that make the variant.
# `-visualROI` keeps the voxels whose split-half reliability is at least 0.3, which
# is about 2000 to 5000 voxels in visual cortex, the only region where short videos
# reliably drive BOLD across repetitions. The GLM-beta variant divides by the
# measured ceiling `VISUAL_ROI_CEILING`; the multimodal variants still return the raw
# correlation, so compare their `score.attrs['raw']`, not the headline numbers.
# `-auditoryROI` keeps early auditory cortex and the planum (Destrieux atlas on
# fsaverage5), where audio features should predict and video features should not.

# GLM-beta variants — one beta per voxel per stimulus per repetition.
benchmark_registry['Lahner2024-fMRI-naturalistic'] = Lahner2024BOLDMoments
benchmark_registry['Lahner2024-fMRI-naturalistic-visualROI'] = partial(
    Lahner2024BOLDMoments, reliability_threshold=0.3, identifier_suffix='-visualROI',
    ceiling=VISUAL_ROI_CEILING)

# TR-resolved variant — predict per-TR BOLD time-series with a clip-mean
# onset-impulse HRF model (each subject-run is a presentation; ridge with
# whole-run held-out blocks, 52 runs per block by default). M12-lite.
# Version 2 (default) selects events by (subject, session, task, run);
# version 1 (defect F6, task omitted) stays reachable for reproducing
# historical scores: load_benchmark(identifier, version=1). The same holds for
# the TR-resolved multimodal registrations below.
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved'] = (
    Lahner2024BOLDMoments_timeresolved)
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-visualROI'] = partial(
    Lahner2024BOLDMoments_timeresolved, reliability_threshold=0.3,
    identifier_suffix='-timeresolved-visualROI')
# `-improved`: per-run motion regression (9 fmriprep confounds, residuals replace the
# BOLD before the z-score) and within-subject CV (contiguous blocks of 10 runs held
# out inside each participant, per-voxel r averaged across participants). The ROI
# raw r moved by +0.001 to +0.016 and the whole-cortex r fell by about half.
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-improved'] = partial(
    Lahner2024BOLDMoments_timeresolved, identifier_suffix='-timeresolved-improved',
    apply_motion_regression=True, cv_mode='within_subject',
    within_subject_n_held_out=10)
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-improved-visualROI'] = partial(
    Lahner2024BOLDMoments_timeresolved, reliability_threshold=0.3,
    identifier_suffix='-timeresolved-improved-visualROI',
    apply_motion_regression=True, cv_mode='within_subject',
    within_subject_n_held_out=10)

# Multimodal A+V variant — first benchmark to exercise the unified
# multi-region recording API + per-modality dispatch on real fMRI.
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal'] = (
    Lahner2024BOLDMoments_multimodal)
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal-visualROI'] = partial(
    Lahner2024BOLDMoments_multimodal, reliability_threshold=0.3,
    identifier_suffix='-multimodal-visualROI')
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal-auditoryROI'] = partial(
    Lahner2024BOLDMoments_multimodal, identifier_suffix='-multimodal-auditoryROI',
    voxel_mask_fn=build_auditory_mask)

# TR-resolved multimodal — clip-mean features placed at clip-onset TRs,
# HRF-convolved, per-voxel ridge against per-TR BOLD time series.
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-multimodal'] = (
    Lahner2024BOLDMoments_timeresolved_multimodal)
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-multimodal-visualROI'] = partial(
    Lahner2024BOLDMoments_timeresolved_multimodal, reliability_threshold=0.3,
    identifier_suffix='-timeresolved-multimodal-visualROI')
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-multimodal-auditoryROI'] = partial(
    Lahner2024BOLDMoments_timeresolved_multimodal,
    identifier_suffix='-timeresolved-multimodal-auditoryROI',
    voxel_mask_fn=build_auditory_mask)
