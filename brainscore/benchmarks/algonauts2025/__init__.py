"""Algonauts 2025 challenge benchmark — Courtois NeuroMod fMRI on movies.

Three modalities (audio + visual + transcript), four subjects, predicts
1000-parcel Schaefer time series at TR=1.49 s. Held-out test on Friends
S7 (in-distribution) and a 2-h OOD movie set.

Status: scaffolding only. Data must be downloaded on EC2 first via
``unified/scripts/download_algonauts_data.sh``. See README.md.
"""
from brainscore import benchmark_registry

from .benchmark import (
    Algonauts2025Friends,
    Algonauts2025FriendsS7,
    Algonauts2025OOD,
)


# Per-subject benchmark factories — Algonauts trains/scores per subject.
# Keep one entry per (split, subject) so the scoring harness can run a
# subject's data through whatever model is being evaluated.
for sub in (1, 2, 3, 5):
    benchmark_registry[f'Algonauts2025-friends-sub{sub:02d}'] = (
        lambda s=sub: Algonauts2025Friends(subject=s))
    benchmark_registry[f'Algonauts2025-friends-s7-sub{sub:02d}'] = (
        lambda s=sub: Algonauts2025FriendsS7(subject=s))
    benchmark_registry[f'Algonauts2025-ood-sub{sub:02d}'] = (
        lambda s=sub: Algonauts2025OOD(subject=s))
