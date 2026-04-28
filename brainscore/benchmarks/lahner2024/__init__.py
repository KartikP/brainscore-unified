from brainscore import benchmark_registry
from .benchmark import Lahner2024BOLDMoments, Lahner2024BOLDMoments_visualROI
from .benchmark_timeresolved import (
    Lahner2024BOLDMoments_timeresolved,
    Lahner2024BOLDMoments_timeresolved_visualROI,
    Lahner2024BOLDMoments_timeresolved_improved,
    Lahner2024BOLDMoments_timeresolved_improved_visualROI,
)

# Existing GLM-beta variants — one beta per voxel per stimulus per repetition.
benchmark_registry['Lahner2024-fMRI-naturalistic'] = Lahner2024BOLDMoments
benchmark_registry['Lahner2024-fMRI-naturalistic-visualROI'] = (
    Lahner2024BOLDMoments_visualROI)

# TR-resolved variant — predict per-TR BOLD time-series via continuous-time
# encoding (each subject-run is a presentation; leave-one-run-out ridge with
# HRF-convolved per-stimulus features). M12-lite. Loading raises until
# prepare_timeresolved_assembly.py uploads the assembly + events to S3 and
# the resulting (version_id, sha1) are pasted into benchmark_timeresolved.py.
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved'] = (
    Lahner2024BOLDMoments_timeresolved)
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-visualROI'] = (
    Lahner2024BOLDMoments_timeresolved_visualROI)
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-improved'] = (
    Lahner2024BOLDMoments_timeresolved_improved)
benchmark_registry['Lahner2024-fMRI-naturalistic-timeresolved-improved-visualROI'] = (
    Lahner2024BOLDMoments_timeresolved_improved_visualROI)
