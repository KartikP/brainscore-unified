"""Algonauts 2025 challenge benchmark: Courtois NeuroMod fMRI on movies.

Four participants; the benchmark predicts 1000-parcel Schaefer time series at
TR = 1.49 s. Each registered identifier uses all released fMRI of one
participant (Friends seasons 1 to 6 and Movie10). The held-out Friends S7 and
the out-of-distribution films have no released fMRI; they are not registered
and serve prediction export through ``brainscore.tools.algonauts2025_predictions``.
Prepare the data first with the data plugin (``brainscore/data/algonauts2025``).
"""
from functools import partial

from brainscore import benchmark_registry


def _load(**configuration):
    from .benchmark import Algonauts2025
    return Algonauts2025(**configuration)


# One benchmark per participant: Algonauts trains and scores per participant.
# Only the split with measured fMRI is registered. The held-out splits
# (Friends S7, OOD) have no target data; build them directly for prediction
# export: brainscore.tools.algonauts2025_predictions.generate_predictions(...).
benchmark_registry['Algonauts2025-friends-sub01'] = partial(
    _load, participant=1, split='friends', identifier='Algonauts2025-friends-sub01')
benchmark_registry['Algonauts2025-friends-sub02'] = partial(
    _load, participant=2, split='friends', identifier='Algonauts2025-friends-sub02')
benchmark_registry['Algonauts2025-friends-sub03'] = partial(
    _load, participant=3, split='friends', identifier='Algonauts2025-friends-sub03')
benchmark_registry['Algonauts2025-friends-sub05'] = partial(
    _load, participant=5, split='friends', identifier='Algonauts2025-friends-sub05')

# The same benchmark with the movie shown as one 1.49 s clip per scan instead of one
# frame (channel vision:video). No sound or speech. All released fMRI of participant 1;
# participants 2, 3 and 5 have no clip form.
benchmark_registry['Algonauts2025-friends-sub01-video'] = partial(
    _load, participant=1, channels=('vision:video',), identifier='Algonauts2025-friends-sub01-video')

# The same benchmark with the clip of each scan on vision:video and the sound of the
# same 1.49 s on audio, in one row. All released fMRI of participant 1 only.
benchmark_registry['Algonauts2025-friends-sub01-video-audio'] = partial(
    _load, participant=1, channels=('vision:video', 'audio'),
    identifier='Algonauts2025-friends-sub01-video-audio')
