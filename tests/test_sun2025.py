"""Sun data separation, scientific identity, and one-presentation UMI behavior."""
import json
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

import brainscore
from brainscore_core import BrainScoreModel
from brainscore_core.selection import LayerSelector
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly, NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.experiments import Experiment, RecordBehavior, RecordNeural
from brainscore.experiments.sun2025 import analyze, make_protocol
from brainscore.experiments.sun2025.analysis import unflip_vector
from brainscore.experiments.sun2025.experiment import parse_report

def stimuli():
    data = StimulusSet([dict(stimulus_id=f'a-f{flip}', base_id='a', scene=1, modulation=1,
                            flip=flip, video_path=f'{flip}.avi') for flip in (1, 2, 3, 4)])
    data.identifier = 'sun-fixture'
    data.stimulus_paths = dict(zip(data.stimulus_id, data.video_path))
    return data


def reference():
    return BehavioralAssembly(np.array([[[3., -4.], [3., -4.], [3., -4.]]]),
                              dims=['presentation', 'participant', 'axis'],
                              coords={'stimulus_id': ('presentation', ['a']),
                                      'scene': ('presentation', [1]),
                                      'participant': [1, 2, 3], 'axis': ['x', 'y']})


def report(direction, speed=5):
    return json.dumps({'direction_deg': direction, 'speed_px_per_frame': speed})


def responses(replies=None):
    angle = np.degrees(np.arctan2(4, 3))
    replies = replies if replies is not None else [
        report(angle), report(180-angle), report(360-angle), report(180+angle)]
    return BehavioralAssembly(np.array(replies), dims=['presentation'],
                              coords={'stimulus_id': ('presentation', [f'a-f{i}' for i in (1, 2, 3, 4)]),
                                      'scene': ('presentation', [1] * 4),
                                      'raw_answer': ('presentation', replies)})


@pytest.mark.parametrize('bad', [
    '', 'left', chr(96)*3 + 'json\n{}\n' + chr(96)*3, '[]', 'null',
    '{"direction_deg": 1, "speed_px_per_frame": -1}',
    '{"direction_deg": 360, "speed_px_per_frame": 1}',
    '{"direction_deg": true, "speed_px_per_frame": 1}',
    '{"direction_deg": NaN, "speed_px_per_frame": 1}',
    '{"direction_deg": 2, "speed_px_per_frame": Infinity}',
    '{"direction_deg": 1, "speed_px_per_frame": 1, "extra": 2}',
    '{"direction_deg": 1, "direction_deg": 2, "speed_px_per_frame": 1}',
])
def test_invalid_motion_report_is_not_repaired(bad):
    assert parse_report(bad)[1]
    assert parse_report(bad)[0] is None


def test_direction_speed_and_flip_coordinates():
    assert unflip_vector(90, 3, 1) == pytest.approx((0, -3))
    assert unflip_vector(90, 3, 3) == pytest.approx((0, 3))
    assert unflip_vector(0, 3, 2) == pytest.approx((-3, 0))
    assert parse_report(report(0, 0))[1] is None


def test_single_json_fence_is_explicitly_supported_without_changing_values():
    from brainscore.experiments.sun2025.experiment import report_format
    fence = chr(96) * 3
    raw = fence + 'json\n' + report(45, 2.5) + '\n' + fence
    assert parse_report(raw) == ({'direction_deg': 45, 'speed_px_per_frame': 2.5}, None)
    assert report_format(raw)[1] == 'json_code_fence'
    assert parse_report('Here is my answer:\n' + raw)[1]
    assert parse_report(raw + '\n' + raw)[1]
    behavior = responses([fence + 'json\n' + answer + '\n' + fence for answer in responses().values])
    trials, _, summary = analyze(behavior, stimuli(), reference())
    assert summary['json_code_fence_answers'] == 4
    assert summary['mean_endpoint_error_px_per_frame'] < 1e-12
    assert trials.raw_answer.iloc[0].startswith(fence)


def test_comparison_is_identity_aligned_and_unflips_before_averaging():
    trials, comparison, summary = analyze(responses().isel(presentation=[3, 1, 0, 2]),
                                          stimuli(), reference())
    assert summary['complete_groups'] == 1
    assert summary['mean_endpoint_error_px_per_frame'] < 1e-12
    assert summary['mean_speed_error_px_per_frame'] < 1e-12
    assert summary['mean_angular_error_deg'] < 1e-5
    assert len(comparison) == 3 and trials.valid.all()


def test_one_bad_flip_invalidates_its_comparison_group():
    behavior = responses(['bad'] + list(responses().values[1:]))
    trials, comparison, summary = analyze(behavior, stimuli(), reference())
    assert summary['valid_answers'] == 3
    assert summary['complete_groups'] == 0
    assert summary['mean_endpoint_error_px_per_frame'] is None
    assert not comparison.valid.any()
    assert trials.raw_answer.iloc[0] == 'bad'


def test_missing_or_duplicate_trials_fail():
    with pytest.raises(ValueError, match='every selected stimulus'):
        analyze(responses().isel(presentation=[0, 1, 2]), stimuli(), reference())
    with pytest.raises(ValueError, match='every selected stimulus'):
        analyze(responses().isel(presentation=[0, 1, 2, 2]), stimuli(), reference())


def test_zero_speed_has_no_defined_direction():
    behavior = responses([report(0, 0)] * 4)
    _, _, summary = analyze(behavior, stimuli(), reference())
    assert summary['mean_angular_error_deg'] is None
    assert summary['angular_comparisons'] == 0
    assert summary['mean_speed_error_px_per_frame'] == 5


class Net:
    def __init__(self):
        self.passes, self.hooks = 0, []

    def forward(self, text):
        self.passes += 1
        for hook in self.hooks:
            hook(np.full(3, len(text), dtype=float))


class Extractor:
    identifier = backbone_id = 'sun-test-extractor'

    def __init__(self, net):
        self.net = net

    def __call__(self, stimuli, layers, **kwargs):
        raise AssertionError('The reply must own the only forward pass')

    @contextmanager
    def record(self, layers):
        seen = []

        class Recording:
            def take(self):
                values = seen.pop(0)
                return NeuroidAssembly(values[None], dims=['presentation', 'neuroid'], coords={
                    'stimulus_id': ('presentation', ['?']),
                    'neuroid_id': ('neuroid', ['early.0', 'early.1', 'early.2']),
                    'layer': ('neuroid', ['early'] * 3)})
        self.net.hooks.append(seen.append)
        try:
            yield Recording()
        finally:
            self.net.hooks.remove(seen.append)


def test_both_recorders_use_one_presentation_and_keep_invalid_reply(tmp_path):
    net = Net()
    seen = []

    def reply(prompt):
        seen.append(prompt.stimulus_id)
        net.forward(prompt.text())
        return 'unparseable' if prompt.stimulus_id == 'a-f2' else report(90)

    subject = BrainScoreModel(identifier='video-test', extractor={'video': Extractor(net)}, reply=reply,
                              region_layer_map={'V1': LayerSelector('early', modality='video')})
    run = Experiment(subject=subject, protocol=make_protocol(stimuli()),
                     tools=[RecordBehavior(), RecordNeural(['V1'])], output_dir=tmp_path / 'run')
    run.validate()
    result = run.run()
    assert net.passes == 4
    assert sorted(seen) == sorted(stimuli().stimulus_id)
    assert result.neural('V1').sizes['presentation'] == 4
    behavior = result.behavior(measure='text')
    trials, _, summary = analyze(behavior, stimuli(), reference())
    assert summary['valid_answers'] == 3
    assert 'unparseable' in list(trials.raw_answer)


def test_still_only_subject_is_incompatible_before_running(tmp_path):
    net = Net()
    subject = BrainScoreModel(identifier='still-only', extractor={'vision': Extractor(net)},
                              reply=lambda prompt: report(0))
    with pytest.raises(ValueError):
        Experiment(subject=subject, protocol=make_protocol(stimuli()), tools=[RecordBehavior()],
                   output_dir=tmp_path / 'no-run').validate()
    assert net.passes == 0 and not (tmp_path / 'no-run').exists()


def test_dataset_registration_is_owned_by_data_plugin():
    from brainscore.data.preparation import data_builder_registry
    identifier = 'Sun2025-second-order-motion'
    assert identifier in data_builder_registry
    assert brainscore.data_registry[identifier].__module__ == 'brainscore.data.sun2025'
    assert brainscore.stimulus_set_registry[identifier].__module__ == 'brainscore.data.sun2025'


def test_protocol_declares_and_emits_native_clips():
    from brainscore.experiments.runner import Repetition
    protocol = make_protocol(stimuli())
    protocol.accept_requests({'behavior'})
    assert protocol.input_channels == frozenset({'vision:clip'})
    with protocol.session_factory(Repetition('default', 0)) as session:
        events = []
        while (event := session.next_input()) is not None:
            events.append(event)
    assert len(events) == 4
    assert all(event.channel == 'vision:clip' and event.t_ms is None for event in events)


def test_prepared_checksum_mismatch_is_rejected(tmp_path):
    from brainscore.data.sun2025.data import load_prepared
    from brainscore.data.sun2025.data_packaging.prepare import checksum
    path = tmp_path / 'human-responses.json'
    path.write_text('[]')
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'dataset': 'Sun2025-second-order-motion', 'schema_version': 1,
        'files': {'human-responses.json': checksum(path)}}))
    path.write_text('[1]')
    with pytest.raises(ValueError, match='checksum'):
        load_prepared(tmp_path)


@pytest.mark.slow
def test_real_prepared_source_and_no_answer_leakage():
    root = os.environ.get('SUN2025_PREPARED')
    if root is None:
        pytest.skip('Set SUN2025_PREPARED for real-source validation')
    from brainscore.data.sun2025.data import load_prepared
    from brainscore.model_helpers.vlm_video_wrapper import read_clip
    data = brainscore.load_stimulus_set('Sun2025-second-order-motion', root=root)
    human = brainscore.load_dataset('Sun2025-second-order-motion', root=root)
    _, manifest = load_prepared(root)
    assert len(data) == len(manifest['scenes']) * 7 * 4
    assert human.shape == (len(manifest['scenes']) * 7, 3, 2)
    assert set(data.columns) == {'stimulus_id', 'base_id', 'scene', 'modulation', 'flip', 'video_path'}
    for path in data.video_path:
        frames, fps = read_clip(path)
        assert frames.shape == (16, 600, 600, 3) and fps == 30
        assert np.all(frames[:, 0, 0] == 128)


def test_human_row_order_is_verified_from_archived_ground_truth():
    from brainscore.data.sun2025.data_packaging.prepare import reference_rows
    table = pd.DataFrame({'MovI': range(1, 41), 'flowU': range(1, 41), 'flowV': [1] * 40})
    participant = pd.DataFrame({'Avg_gt_X': np.arange(1, 41) * 1.5 / np.sqrt(2),
                                'Avg_gt_Y': np.full(40, 1.5 / np.sqrt(2))})
    for mod in range(1, 8):
        participant[f'Mod{mod}_avg_res_X'] = 1
        participant[f'Mod{mod}_avg_res_Y'] = 2
    assert len(reference_rows(table, [participant] * 3, [1])) == 21
    with pytest.raises(ValueError, match='ordering'):
        reference_rows(table, [participant.iloc[::-1]] * 3, [1])


def test_source_image_boundary_shows_display_background(tmp_path):
    from types import SimpleNamespace
    from brainscore.data.sun2025.data_packaging.prepare import render_clip
    from brainscore.model_helpers.vlm_video_wrapper import read_clip
    frame = np.zeros((1024, 1024, 3), dtype=np.uint8)
    row = SimpleNamespace(picX=150, picY=500, fliptype=1, R=0., G=0., B=0.)
    path = tmp_path / 'edge.avi'
    render_clip([frame] * 16, row, path)
    frames, fps = read_clip(path)
    assert np.all(frames[8, 300, 20] == 128)
    assert np.all(frames[8, 300, 120] < 5)
    assert fps == 30
