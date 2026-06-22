"""Unit tests for VideoWrapper using a mock video model.

We avoid real video models here (they're 100+ MB and can't be downloaded
reliably from HF in CI). A tiny hand-crafted PyTorch module does the job
to validate the infrastructure: frame sampling → preprocessing → batched
forward → hook → (presentation, time_bin, neuroid) assembly.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_mock_video_model(n_features: int = 8):
    """Tiny model that takes (B, T, C, H, W) and outputs (B, T, n_features).

    Has a named sub-module 'main_block' that produces the hookable output.
    """
    import torch
    import torch.nn as nn

    class MockBlock(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.linear = nn.Linear(3 * 8 * 8, n_features)

        def forward(self, x):
            # x: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            flat = x.reshape(B * T, C * H * W)
            out = self.linear(flat)  # (B*T, n_features)
            return out.reshape(B, T, -1)

    class MockVideoModel(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.main_block = MockBlock(n_features)

        def forward(self, x):
            return self.main_block(x)

    torch.manual_seed(0)
    return MockVideoModel(n_features)


def _mock_preprocess(frames):
    """Turn list of HxWx3 numpy arrays into (T, C, H, W) float32 tensor
    at resolution 8x8 (small for tests)."""
    import torch
    out = []
    for frame in frames:
        # Resize to 8x8 by naive subsampling (keep it dependency-free)
        h_stride = max(1, frame.shape[0] // 8)
        w_stride = max(1, frame.shape[1] // 8)
        small = frame[::h_stride, ::w_stride][:8, :8]
        if small.shape[:2] != (8, 8):
            small = np.pad(small, (
                (0, 8 - small.shape[0]), (0, 8 - small.shape[1]), (0, 0)),
                mode='edge')
        out.append(small.astype(np.float32) / 255.0)
    stacked = np.stack(out, axis=0)  # (T, H, W, C)
    # Reorder to (T, C, H, W)
    stacked = stacked.transpose(0, 3, 1, 2)
    return torch.from_numpy(stacked)


def _make_tiny_video(path: Path, n_frames: int = 10, fps: float = 5.0,
                     size: tuple = (16, 16)):
    """Write a tiny MP4 with random frames for testing."""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        pytest.skip("cv2 VideoWriter could not open mp4v codec")
    rng = np.random.default_rng(abs(hash(str(path))) % (2**32))
    for i in range(n_frames):
        # BGR random frame
        frame = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def video_files(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f'vid{i}.mp4'
        _make_tiny_video(p, n_frames=10, fps=5.0)
        paths.append(str(p))
    return paths


@pytest.fixture
def video_stimulus_set(video_files):
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    df = pd.DataFrame({
        'stimulus_id': [f'v{i}' for i in range(len(video_files))],
        'video_path': video_files,
    })
    stim = StimulusSet(df)
    stim.identifier = 'test_video_set'
    stim.stimulus_paths = dict(zip(df['stimulus_id'], df['video_path']))
    return stim


class TestVideoWrapperBasics:
    def test_returns_3d_assembly(self, video_stimulus_set):
        from brainscore.model_helpers.video_wrapper import VideoWrapper
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

        model = _make_mock_video_model(n_features=8)
        wrapper = VideoWrapper(
            model=model,
            preprocessing=_mock_preprocess,
            identifier='mock-vid',
            num_frames=4,
            batch_size=2,
        )
        result = wrapper(video_stimulus_set, layers=['main_block'])
        assert isinstance(result, NeuroidAssembly)
        assert result.dims == ('presentation', 'time_bin', 'neuroid')

    def test_shape(self, video_stimulus_set):
        from brainscore.model_helpers.video_wrapper import VideoWrapper
        model = _make_mock_video_model(n_features=8)
        wrapper = VideoWrapper(
            model=model,
            preprocessing=_mock_preprocess,
            identifier='mock-vid',
            num_frames=4,
            batch_size=2,
        )
        result = wrapper(video_stimulus_set, layers=['main_block'])
        # 3 videos, 4 frames each (T_out matches input T for this mock model),
        # 8 features per time step
        assert result.shape == (3, 4, 8)

    def test_stimulus_ids_preserved(self, video_stimulus_set):
        from brainscore.model_helpers.video_wrapper import VideoWrapper
        model = _make_mock_video_model(n_features=8)
        wrapper = VideoWrapper(
            model=model,
            preprocessing=_mock_preprocess,
            identifier='mock-vid',
            num_frames=4,
        )
        result = wrapper(video_stimulus_set, layers=['main_block'])
        stim_level = list(result.indexes['presentation'].get_level_values('stimulus_id'))
        assert stim_level == ['v0', 'v1', 'v2']

    def test_time_bin_centers_populated(self, video_stimulus_set):
        from brainscore.model_helpers.video_wrapper import VideoWrapper
        model = _make_mock_video_model(n_features=8)
        wrapper = VideoWrapper(
            model=model,
            preprocessing=_mock_preprocess,
            identifier='mock-vid',
            num_frames=4,
        )
        result = wrapper(video_stimulus_set, layers=['main_block'])
        centers = list(result.indexes['time_bin'].get_level_values('time_bin_center_ms'))
        assert len(centers) == 4
        # With 10 frames at 5 fps, duration ~2000 ms
        # 4 time steps evenly spread → centers roughly [250, 750, 1250, 1750]
        assert centers[0] < centers[-1]
        assert centers[0] > 0
        assert centers[-1] < 2500

    def test_neuroid_coords_carry_layer_label(self, video_stimulus_set):
        from brainscore.model_helpers.video_wrapper import VideoWrapper
        model = _make_mock_video_model(n_features=8)
        wrapper = VideoWrapper(
            model=model,
            preprocessing=_mock_preprocess,
            identifier='mock-vid',
            num_frames=4,
        )
        result = wrapper(video_stimulus_set, layers=['main_block'])
        layer_labels = list(
            result.indexes['neuroid'].get_level_values('layer'))
        assert set(layer_labels) == {'main_block'}

    def test_time_bin_start_end_populated(self, video_stimulus_set):
        """VideoWrapper emits the canonical time_bin_start_ms / time_bin_end_ms
        coords (matching Text/Audio wrappers), bracketing each center."""
        from brainscore.model_helpers.video_wrapper import VideoWrapper
        model = _make_mock_video_model(n_features=8)
        wrapper = VideoWrapper(
            model=model,
            preprocessing=_mock_preprocess,
            identifier='mock-vid',
            num_frames=4,
        )
        result = wrapper(video_stimulus_set, layers=['main_block'])
        idx = result.indexes['time_bin']
        centers = list(idx.get_level_values('time_bin_center_ms'))
        starts = list(idx.get_level_values('time_bin_start_ms'))
        ends = list(idx.get_level_values('time_bin_end_ms'))
        assert len(starts) == len(ends) == len(centers)
        # Each bin brackets its center, is non-empty, starts at/after 0,
        # and bins are contiguous (end[i] == start[i+1] under midpoints).
        assert starts[0] >= 0.0
        for i in range(len(centers)):
            assert starts[i] <= centers[i] <= ends[i]
            assert ends[i] > starts[i]
        for i in range(len(centers) - 1):
            assert ends[i] == pytest.approx(starts[i + 1])


class TestVideoWrapperTemporalInvariance:
    def test_different_frame_orders_give_different_activations(
            self, video_stimulus_set):
        """A video-native wrapper MUST preserve temporal order — reversing
        the video should give different activations. (The frame-aggregation
        path in the Lahner2024 benchmark cannot do this; that's the whole
        point of VideoWrapper.)
        """
        from brainscore.model_helpers.video_wrapper import VideoWrapper

        model = _make_mock_video_model(n_features=8)
        wrapper = VideoWrapper(
            model=model,
            preprocessing=_mock_preprocess,
            identifier='mock-vid',
            num_frames=4,
        )

        forward_result = wrapper(video_stimulus_set, layers=['main_block'])

        # Build a reversed sampler and a new wrapper using it
        def reversed_sampler(video_path, num_frames=0, target_fps=None):
            from brainscore.model_helpers.video_wrapper import default_uniform_frame_sampler
            frames = default_uniform_frame_sampler(
                video_path, num_frames=num_frames, target_fps=target_fps)
            return list(reversed(frames))

        wrapper_rev = VideoWrapper(
            model=model,
            preprocessing=_mock_preprocess,
            identifier='mock-vid-reversed',
            num_frames=4,
            frame_sampler=reversed_sampler,
        )
        reverse_result = wrapper_rev(video_stimulus_set, layers=['main_block'])

        # Per video, the activations should DIFFER at the corresponding
        # time steps (because the input frames were re-ordered).
        f = forward_result.values
        r = reverse_result.values
        assert not np.allclose(f, r), (
            "Forward and reversed frame orders produced identical activations —"
            " VideoWrapper is not temporally sensitive!")


class TestVideoWrapperBatching:
    def test_batch_size_1_vs_batch_size_3(self, video_stimulus_set):
        """Scoring with batch_size=1 and batch_size=3 should produce
        the same output (modulo numerical noise)."""
        from brainscore.model_helpers.video_wrapper import VideoWrapper
        model1 = _make_mock_video_model(n_features=8)
        model3 = _make_mock_video_model(n_features=8)  # same seed, same weights
        w1 = VideoWrapper(model=model1, preprocessing=_mock_preprocess,
                          identifier='mock', num_frames=4, batch_size=1)
        w3 = VideoWrapper(model=model3, preprocessing=_mock_preprocess,
                          identifier='mock', num_frames=4, batch_size=3)
        r1 = w1(video_stimulus_set, layers=['main_block'])
        r3 = w3(video_stimulus_set, layers=['main_block'])
        np.testing.assert_allclose(r1.values, r3.values, atol=1e-5)


class TestEdgesFromCenters:
    """Unit tests for the bin-center → (start, end) edge derivation."""

    def test_uniform_grid_is_exact(self):
        """For the uniform default grid, midpoint edges recover the true
        bin boundaries exactly: centers [0.5, 1.5, 2.5, 3.5]*step →
        edges [0,1],[1,2],[2,3],[3,4]*step."""
        from brainscore.model_helpers.video_wrapper import _edges_from_centers
        centers = [250.0, 750.0, 1250.0, 1750.0]  # step=500, n=4
        starts, ends = _edges_from_centers(centers)
        assert starts == pytest.approx([0.0, 500.0, 1000.0, 1500.0])
        assert ends == pytest.approx([500.0, 1000.0, 1500.0, 2000.0])

    def test_single_bin_spans_full_clip(self):
        from brainscore.model_helpers.video_wrapper import _edges_from_centers
        starts, ends = _edges_from_centers([1000.0])  # center = duration/2
        assert starts == [0.0]
        assert ends == pytest.approx([2000.0])

    def test_empty(self):
        from brainscore.model_helpers.video_wrapper import _edges_from_centers
        assert _edges_from_centers([]) == ([], [])

    def test_non_uniform_is_monotonic_and_contiguous(self):
        from brainscore.model_helpers.video_wrapper import _edges_from_centers
        centers = [100.0, 200.0, 500.0]  # widening gaps
        starts, ends = _edges_from_centers(centers)
        # contiguous: end[i] == start[i+1]
        for i in range(len(centers) - 1):
            assert ends[i] == pytest.approx(starts[i + 1])
        # each bin brackets its center, non-empty, start clamped at 0
        assert starts[0] >= 0.0
        for i in range(len(centers)):
            assert starts[i] <= centers[i] <= ends[i]
            assert ends[i] > starts[i]


# ---- long-clip temporal-context chunking ----

def _long_video_stim(tmp_path, n_frames=60, fps=5.0):
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    p = tmp_path / 'long.mp4'
    _make_tiny_video(p, n_frames=n_frames, fps=fps)
    df = pd.DataFrame({'stimulus_id': ['L0'], 'video_path': [str(p)]})
    stim = StimulusSet(df)
    stim.identifier = 'long_set'
    stim.stimulus_paths = {'L0': str(p)}
    return stim


def _chunk_wrapper(**kw):
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    return VideoWrapper(model=_make_mock_video_model(8), preprocessing=_mock_preprocess,
                        identifier='mock-vid', num_frames=4, **kw)


class TestVideoWrapperChunking:
    def test_block_chunking_stitches_clip_time(self, tmp_path):
        stim = _long_video_stim(tmp_path)  # 60 frames @ 5fps = 12000 ms
        r = _chunk_wrapper(context_window_ms=4000)(stim, layers=['main_block'])
        assert r.shape == (1, 12, 8)  # 3 block windows * 4 frames
        assert r['time_bin_start_ms'].values.min() >= 0
        assert float(r['time_bin_end_ms'].values.max()) == pytest.approx(12000, abs=1500)
        assert (np.diff(r['time_bin_center_ms'].values) > 0).all()  # monotone clip time

    def test_fail_fast_on_long_clip_when_unconfigured(self, tmp_path):
        stim = _long_video_stim(tmp_path)  # 12000 ms, no context_window_ms
        with pytest.raises(ValueError, match='max_clip_ms'):
            _chunk_wrapper(max_clip_ms=5000)(stim, layers=['main_block'])

    def test_causal_one_feature_per_stride(self, tmp_path):
        stim = _long_video_stim(tmp_path)
        r = _chunk_wrapper(context_window_ms=4000, context_strategy='causal')(
            stim, layers=['main_block'])
        assert r.shape == (1, 3, 8)  # output times 4000, 8000, 12000


def test_pad_out_of_bound_strategies():
    from brainscore.model_helpers.video_wrapper import pad_out_of_bound
    f = [np.full((2, 2, 3), 10, np.uint8), None, None]
    oob = [False, True, True]
    assert (pad_out_of_bound(f, oob, 'repeat')[2] == 10).all()   # nearest valid
    assert (pad_out_of_bound(f, oob, 'black')[1] == 0).all()
    assert (pad_out_of_bound(f, oob, 'gray')[1] == 128).all()
    with pytest.raises(ValueError):
        pad_out_of_bound([None], [True], 'repeat')               # no valid frame
