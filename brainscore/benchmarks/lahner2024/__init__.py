from brainscore import benchmark_registry
from .benchmark import Lahner2024BOLDMoments, Lahner2024BOLDMoments_visualROI
from .benchmark_timeresolved import (
    Lahner2024BOLDMoments_timeresolved,
    Lahner2024BOLDMoments_timeresolved_visualROI,
    Lahner2024BOLDMoments_timeresolved_improved,
    Lahner2024BOLDMoments_timeresolved_improved_visualROI,
)
from .benchmark_multimodal import (
    Lahner2024BOLDMoments_multimodal,
    Lahner2024BOLDMoments_multimodal_visualROI,
    Lahner2024BOLDMoments_multimodal_auditoryROI,
)
from .benchmark_timeresolved_multimodal import (
    Lahner2024BOLDMoments_timeresolved_multimodal,
    Lahner2024BOLDMoments_timeresolved_multimodal_visualROI,
    Lahner2024BOLDMoments_timeresolved_multimodal_auditoryROI,
)

# Existing GLM-beta variants — one beta per voxel per stimulus per repetition.
benchmark_registry['Lahner2024-fMRI-naturalistic'] = Lahner2024BOLDMoments
benchmark_registry['Lahner2024-fMRI-naturalistic-visualROI'] = (
    Lahner2024BOLDMoments_visualROI)

# TR-resolved variant — predict per-TR BOLD time-series with a clip-mean
# onset-impulse HRF model (each subject-run is a presentation; ridge with
# whole-run held-out blocks, 52 runs per block by default). M12-lite.
# Version 2 (default) selects events by (subject, session, task, run);
# version 1 (defect F6, task omitted) stays reachable for reproducing
# historical scores: load_benchmark(identifier, version=1). The same holds for
# the TR-resolved multimodal registrations below.
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved'] = (
    Lahner2024BOLDMoments_timeresolved)
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-visualROI'] = (
    Lahner2024BOLDMoments_timeresolved_visualROI)
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-improved'] = (
    Lahner2024BOLDMoments_timeresolved_improved)
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-improved-visualROI'] = (
    Lahner2024BOLDMoments_timeresolved_improved_visualROI)

# Multimodal A+V variant — first benchmark to exercise the unified
# multi-region recording API + per-modality dispatch on real fMRI.
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal'] = (
    Lahner2024BOLDMoments_multimodal)
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal-visualROI'] = (
    Lahner2024BOLDMoments_multimodal_visualROI)
benchmark_registry['Lahner2024-fMRI-naturalistic-multimodal-auditoryROI'] = (
    Lahner2024BOLDMoments_multimodal_auditoryROI)

# TR-resolved multimodal — clip-mean features placed at clip-onset TRs,
# HRF-convolved, per-voxel ridge against per-TR BOLD time series.
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-multimodal'] = (
    Lahner2024BOLDMoments_timeresolved_multimodal)
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-multimodal-visualROI'] = (
    Lahner2024BOLDMoments_timeresolved_multimodal_visualROI)
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-multimodal-auditoryROI'] = (
    Lahner2024BOLDMoments_timeresolved_multimodal_auditoryROI)
