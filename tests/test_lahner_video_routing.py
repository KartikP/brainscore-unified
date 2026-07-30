"""Routing regression: native-video models must not be misrouted to the
frame-aggregation path after channel unification canonicalizes video -> vision.

Task 6 of the Group B EC2 smoke found that vjepa2-vitl (a VideoWrapper model)
crashed in temporal_bin because the benchmark routed on the literal 'video'
modality, which canonicalization removes from supported_modalities. The fix
routes on the RAW input_modalities instead.
"""
from types import SimpleNamespace

from brainscore.benchmarks.lahner2024.benchmark import Lahner2024BOLDMoments


def test_native_video_detected_via_raw_input_modalities():
    # a VideoWrapper model reports supported_modalities={'vision'} (canonicalized)
    # but input_modalities={'video'} (raw) -> must route video_native.
    native = SimpleNamespace(input_modalities={'video'}, supported_modalities={'vision'})
    assert Lahner2024BOLDMoments._is_native_video(native) is True


def test_still_image_model_routes_frame_aggregation():
    still = SimpleNamespace(input_modalities={'vision'}, supported_modalities={'vision'})
    assert Lahner2024BOLDMoments._is_native_video(still) is False


def test_model_without_input_modalities_falls_back_to_frame():
    # legacy adapters / bare candidates: no input_modalities attr -> frame path.
    assert Lahner2024BOLDMoments._is_native_video(object()) is False


def test_old_supported_modalities_check_would_have_misrouted():
    # the exact regression this fixes: the OLD check ('video' in
    # supported_modalities) is False for a native-video model post-canonicalization
    # -> it fell into the frame-aggregation path and crashed temporal_bin.
    native = SimpleNamespace(input_modalities={'video'}, supported_modalities={'vision'})
    assert 'video' not in native.supported_modalities          # the trap
    assert Lahner2024BOLDMoments._is_native_video(native)      # the fix routes correctly
