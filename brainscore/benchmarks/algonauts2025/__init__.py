"""Algonauts 2025 challenge benchmark — Courtois NeuroMod fMRI on movies.

Three modalities (audio + visual + transcript), four subjects, predicts
1000-parcel Schaefer time series at TR=1.49 s. The held-out Friends S7
(in-distribution) and the 2-h OOD movie set have no released fMRI; they are
not registered and serve prediction export through ``Algonauts2025(split=...)``.

Status: scaffolding only. Data must be downloaded on EC2 first via
DataLad (see README.md for the exact dataset paths). See README.md.
"""
from brainscore import benchmark_registry

from .benchmark import Algonauts2025


# One benchmark per participant: Algonauts trains and scores per participant.
# Only the split with measured fMRI is registered. The held-out splits
# (Friends S7, OOD) have no target data; build them directly for prediction
# export: Algonauts2025(participant=1, split='friends_s7').generate_predictions(...).
for sub in (1, 2, 3, 5):
    benchmark_registry[f'Algonauts2025-friends-sub{sub:02d}'] = (
        lambda s=sub: Algonauts2025(participant=s, split='friends'))

# The same benchmark with the movie shown as one 1.49 s clip per scan instead of one
# frame (channel vision:video). No sound or speech. Training split of participant 1 only.
benchmark_registry['Algonauts2025-friends-sub01-video'] = (
    lambda: Algonauts2025(participant=1, channels=('vision:video',)))

# The same benchmark with the clip of each scan on vision:video and the sound of the
# same 1.49 s on audio, in one row. Training split of participant 1 only.
benchmark_registry['Algonauts2025-friends-sub01-video-audio'] = (
    lambda: Algonauts2025(participant=1, channels=('vision:video', 'audio')))
