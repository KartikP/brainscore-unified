"""Frame/bin identity and truthful replay of saved activity. Execute remotely."""
import json

import numpy as np
import pytest

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly, BehavioralAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.experiments.sun2025.replay import _frame_bins, _payload, replay_html

pytestmark = pytest.mark.unit


def fixtures():
    stimuli = StimulusSet({'stimulus_id': ['first', 'second'], 'video_path': ['first.avi', 'second.avi']})
    neural = NeuroidAssembly(
        np.array([[[100., 200.], [300., 400.]], [[1., 3.], [2., 3.]]]),
        dims=['presentation', 'time_bin', 'neuroid'],
        coords={'stimulus_id': ('presentation', ['second', 'first']),
                'scene': ('presentation', [2, 1]),
                'time_bin_start_ms': ('time_bin', [0., 100.]),
                'time_bin_end_ms': ('time_bin', [100., 200.]),
                'neuroid_id': ('neuroid', ['a', 'b']),
                'layer': ('neuroid', ['layer', 'layer'])})
    behavior = BehavioralAssembly(
        ['unused', '{"direction_deg": 90, "speed_px_per_frame": 3}'],
        dims=['presentation'],
        coords={'stimulus_id': ('presentation', ['second', 'first']),
                'scene': ('presentation', [2, 1])})
    return stimuli, neural, behavior


@pytest.fixture
def clip(monkeypatch):
    from brainscore.model_helpers import vlm_video_wrapper
    monkeypatch.setattr(vlm_video_wrapper, 'read_clip',
                        lambda path: (np.zeros((4, 10, 10, 3), dtype=np.uint8), 20.))


def test_two_frames_per_patch_and_gaps_stay_missing():
    assert _frame_bins(4, 20., [0, 100], [100, 200]) == [0, 0, 1, 1]
    assert _frame_bins(4, 20., [0, 150], [50, 200]) == [0, -1, -1, 1]


@pytest.mark.parametrize('starts,ends', [([0, 50], [100, 200]), ([100, 0], [200, 100]),
                                       ([0, np.nan], [100, 200]), ([0], [0])])
def test_invalid_recording_times_fail(starts, ends):
    with pytest.raises(ValueError, match='time bins'):
        _frame_bins(4, 20., starts, ends)


def test_payload_aligns_by_identity_and_preserves_raw_activity(clip):
    stimuli, neural, behavior = fixtures()
    original = neural.values.copy()
    payload = _payload(stimuli, neural, 'first', behavior)
    np.testing.assert_array_equal(neural.values, original)
    assert payload['raw_activity'] == [[1., 3.], [2., 3.]]
    assert payload['display_activity'] == [[-1., 0.], [1., 0.]]
    assert payload['frame_bins'] == [0, 0, 1, 1]
    assert payload['report'] == {'direction_deg': 90, 'speed_px_per_frame': 3}
    assert payload['unit_ids'] == ['a', 'b']


def test_static_activity_cannot_be_given_invented_timing(clip):
    stimuli, neural, _ = fixtures()
    with pytest.raises(ValueError, match='cannot invent timing'):
        _payload(stimuli, neural.mean('time_bin'), 'first')


def test_duplicate_recording_is_not_silently_selected(clip):
    stimuli, neural, _ = fixtures()
    with pytest.raises(ValueError, match='exactly one recording'):
        _payload(stimuli, neural.isel(presentation=[1, 1]), 'first')


def test_html_preserves_data_and_does_not_execute_answer_text(clip):
    stimuli, neural, behavior = fixtures()
    raw = '</script><script>alert(1)</script>'
    behavior = BehavioralAssembly([raw, raw], dims=['presentation'], coords={
        'stimulus_id': ('presentation', ['second', 'first']), 'scene': ('presentation', [2, 1])})
    html = replay_html(stimuli, neural, 'first', behavior=behavior)
    assert raw not in html
    encoded = html.split('<script id="replay-data" type="application/json">', 1)[1].split('</script>', 1)[0]
    payload = json.loads(encoded)
    assert payload['raw_answer'] == raw and payload['parse_error']
    assert payload['frame_count'] == 4
    assert '__REPLAY_DATA__' not in html
