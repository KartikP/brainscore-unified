"""Algonauts 2025 challenge benchmark — Courtois NeuroMod fMRI on movies.

Three modalities (audio + visual + transcript), four subjects, predicts
1000-parcel Schaefer time series at TR=1.49 s. Held-out test on Friends
S7 (in-distribution) and a 2-h OOD movie set.

Status: scaffolding only. Data must be downloaded on EC2 first via
DataLad (see README.md for the exact dataset paths). See README.md.
"""
from brainscore import benchmark_registry

from .benchmark import Algonauts2025


# Per-participant benchmarks — Algonauts trains and scores per participant.
# Keep one entry per (split, participant) so the scoring harness can run a
# participant's data through whatever subject is being evaluated.
for sub in (1, 2, 3, 5):
    benchmark_registry[f'Algonauts2025-friends-sub{sub:02d}'] = (
        lambda s=sub: Algonauts2025(participant=s, split='friends'))
    benchmark_registry[f'Algonauts2025-friends-s7-sub{sub:02d}'] = (
        lambda s=sub: Algonauts2025(participant=s, split='friends_s7'))
    benchmark_registry[f'Algonauts2025-ood-sub{sub:02d}'] = (
        lambda s=sub: Algonauts2025(participant=s, split='ood'))

# The same benchmark with the movie shown as one 1.49 s clip per scan instead of one
# frame (channel vision:video). No sound or speech. Training split of participant 1 only.
benchmark_registry['Algonauts2025-friends-sub01-video'] = (
    lambda: Algonauts2025(participant=1, channels=('vision:video',)))

# The same benchmark with the clip of each scan on vision:video and the sound of the
# same 1.49 s on audio, in one row. Training split of participant 1 only.
benchmark_registry['Algonauts2025-friends-sub01-video-audio'] = (
    lambda: Algonauts2025(participant=1, channels=('vision:video', 'audio')))
