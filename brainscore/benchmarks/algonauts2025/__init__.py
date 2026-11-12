"""Algonauts 2025 challenge benchmark — Courtois NeuroMod fMRI on movies.

Three modalities (audio + visual + transcript), four subjects, predicts
1000-parcel Schaefer time series at TR=1.49 s. Held-out test on Friends
S7 (in-distribution) and a 2-h OOD movie set.

Status: scaffolding only. Data must be downloaded on EC2 first via
DataLad (see README.md for the exact dataset paths). See README.md.
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

# The same benchmark with the movie shown as one 1.49 s clip per scan instead of one
# frame (form='video', channel vision:video). No sound or speech yet. Training split of
# participant 1 only.
benchmark_registry['Algonauts2025-friends-sub01-video'] = (
    lambda: Algonauts2025Friends(subject=1, form='video'))

# The same benchmark with the clip of each scan on vision:video and the sound of the
# same 1.49 s on audio, in one row (form='video+audio'). It records neural:IT and
# neural:A1 and scores the two bands with banded ridge. Training split of participant 1
# only.
benchmark_registry['Algonauts2025-friends-sub01-video-audio'] = (
    lambda: Algonauts2025Friends(subject=1, form='video+audio'))

# The video and sound form, but the model answers with every layer it mapped
# (neural:all) and one regression predicts the whole brain from all of them.
benchmark_registry['Algonauts2025-friends-sub01-wholebrain'] = (
    lambda: Algonauts2025Friends(subject=1, form='video+audio', record='all'))
