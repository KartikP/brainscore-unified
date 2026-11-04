"""Lahner2024 shows a subject clips when it declares it takes them (vision:clip),
and frames sampled from each clip otherwise. The route comes from declarations,
never from raw extractor keys."""
from brainscore_core.brainscore_model import BrainScoreModel

from brainscore.benchmarks.lahner2024.benchmark import Lahner2024BOLDMoments


def _extract(stimuli, layers, **kwargs):
    raise AssertionError('routing only')


def test_a_clip_model_is_shown_clips():
    assert Lahner2024BOLDMoments._is_native_video(BrainScoreModel('v', extractor={'video': _extract}))


def test_a_still_model_is_shown_frames():
    assert not Lahner2024BOLDMoments._is_native_video(BrainScoreModel('s', extractor={'vision': _extract}))


def test_a_model_with_both_forms_is_shown_clips():
    both = BrainScoreModel('b', extractor={'vision': _extract, 'video': _extract})
    assert Lahner2024BOLDMoments._is_native_video(both)
