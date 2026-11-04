"""VLMVideoWrapper: whole clips through the model's own video processor, one time bin
per temporal patch, both passively and while another caller runs the model."""
import numpy as np
import pytest
import torch

from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.model_helpers.vlm_video_wrapper import VLMVideoWrapper, read_clip

pytestmark = pytest.mark.unit
cv2 = pytest.importorskip('cv2')


def _write_clip(path, moving):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 4.0, (16, 16))
    for i in range(8):                                     # 2 s at 4 fps
        frame = np.zeros((16, 16, 3), np.uint8)
        frame[:, (2 * i if moving else 0):(2 * i if moving else 0) + 4] = 255
        writer.write(frame)
    writer.release()
    return str(path)


class VideoProcessor:
    """Stand-in for Qwen's: samples at 2 fps, pairs frames into temporal patches of 2x2 positions."""
    fps, temporal_patch_size = 2, 2

    def __call__(self, videos, video_metadata, return_tensors):
        frames = videos[0].astype(np.float32) / 255
        step = int(round(video_metadata[0].fps / self.fps)) if hasattr(video_metadata[0], 'fps') \
            else int(round(video_metadata[0]['fps'] / self.fps))
        sampled = frames[::step]                           # 4 frames -> 2 temporal patches
        t = len(sampled) // self.temporal_patch_size
        patches = []
        for k in range(t):
            pair = sampled[2 * k:2 * k + 2].mean(axis=0)
            for qh in range(2):
                for qw in range(2):
                    block = pair[qh * 8:(qh + 1) * 8, qw * 8:(qw + 1) * 8]
                    patches.append([block.mean(), float(qh), float(qw), 1.0])
        return {'pixel_values_videos': torch.tensor(patches), 'video_grid_thw': torch.tensor([[t, 2, 2]])}


class Processor:
    video_processor = VideoProcessor()


class Tower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(),
                                                               torch.nn.Linear(4, 4))])

    def forward(self, hidden, grid_thw):
        return self.blocks[0](hidden)


def test_a_clip_gives_one_time_bin_per_temporal_patch(tmp_path):
    wrapper = VLMVideoWrapper(Tower(), Processor(), identifier='tiny-video')
    stimuli = StimulusSet({'stimulus_id': ['moving', 'still'],
                           'video_path': [_write_clip(tmp_path / 'm.avi', True),
                                          _write_clip(tmp_path / 's.avi', False)]})
    assembly = wrapper(stimuli, layers=['blocks.0'])
    assert assembly.dims == ('presentation', 'time_bin', 'neuroid')
    assert assembly.shape == (2, 2, 4)
    assert list(assembly['time_bin_start_ms'].values) == [0.0, 1000.0]
    moving, still = np.asarray(assembly.values)
    assert not np.allclose(moving[0], moving[1])           # motion changes the activity over time
    np.testing.assert_allclose(still[0], still[1], atol=1e-5)               # a static clip does not
    assert list(assembly['stimulus_id'].values) == ['moving', 'still']


def test_record_takes_the_clip_another_caller_ran(tmp_path):
    tower, processor = Tower(), Processor()
    wrapper = VLMVideoWrapper(tower, processor, identifier='tiny-video')
    frames, fps = read_clip(_write_clip(tmp_path / 'm.avi', True))
    from brainscore.model_helpers.vlm_video_wrapper import clip_metadata
    inputs = processor.video_processor(videos=[frames], video_metadata=[clip_metadata(frames, fps)],
                                       return_tensors='pt')
    with wrapper.record(['blocks.0']) as recording:
        tower(inputs['pixel_values_videos'], grid_thw=inputs['video_grid_thw'])   # e.g. inside generate
        taken = recording.take()
    passive = wrapper(StimulusSet({'stimulus_id': ['m'], 'video_path': [str(tmp_path / 'm.avi')]}),
                      layers=['blocks.0'])
    np.testing.assert_allclose(taken.values, passive.values, rtol=1e-6)
    assert not tower._forward_pre_hooks and not tower.blocks[0]._forward_hooks
