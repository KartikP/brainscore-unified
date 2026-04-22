"""
VideoWrapper — activations_model for native-temporal video models.

Fourth wrapper class in the unified interface, alongside:
    - PytorchWrapper      (standard image models, batch x C x H x W)
    - TextWrapper         (sequence models, batch x seq_len)
    - VLMVisionWrapper    (VLMs with flattened-patch vision input)
    - VideoWrapper        (this file: video-native models)

A native-temporal video model consumes a ``(batch, T, C, H, W)`` tensor
(or a similar layout) and produces activations that span time — e.g., a
transformer that attends across frames. Examples: VideoMAE, V-JEPA,
TimeSformer, VideoMAEv2.

Unlike the frame-aggregation path used by image models on the Lahner2024
benchmark, this wrapper preserves temporal structure end-to-end:

    raw video (MP4)
       ↓ sample T frames at target fps
       ↓ preprocess → tensor (T, C, H, W)
       ↓ stack B videos → (B, T, C, H, W)
       ↓ model forward, hook layer outputs (B, T_hook, features) or similar
       ↓ map T_hook → video time (ms) via target fps + hook-output rate
       ↓ NeuroidAssembly with dims (presentation, time_bin, neuroid)

The output assembly already has a ``time_bin`` dimension; downstream code
(e.g., ``temporal_bin`` for further aggregation, or a benchmark that
matches against fMRI TRs) consumes it directly.

Design symmetry with other wrappers:
    - Identifier property → used by @store_xarray caching key
    - Accepts a StimulusSet with ``video_path`` column (or list of paths)
    - Takes ``layers`` kwarg; returns NeuroidAssembly
    - No new methods — just the __call__ contract

Design asymmetry (video-specific):
    - Requires a callable ``frame_sampler`` to extract T frames per video
      at a target fps (OpenCV / ffmpeg). Kept configurable rather than
      bundled so brainscore_core stays dependency-free.
    - Requires a callable ``preprocessing`` that takes a list of T frames
      (as numpy HxWx3 arrays) and returns a model-ready tensor.
    - Configurable ``t_to_time_ms_fn`` maps hook-output time index to
      absolute ms within the video, for layers that downsample time
      (e.g., a transformer with patch-wise temporal compression).

Memory: videos are processed one-at-a-time by default (batch_size=1)
because video models have large T×C×H×W tensors. Batch size can be
increased if memory permits.
"""

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuroidAssembly, walk_coords,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from result_caching import store_xarray


logger = logging.getLogger(__name__)


def default_uniform_frame_sampler(
    video_path: Union[str, Path],
    num_frames: int,
    target_fps: Optional[float] = None,
) -> List[np.ndarray]:
    """Sample ``num_frames`` frames uniformly across a video's duration.

    Uses OpenCV (cv2). If ``target_fps`` is provided, sampling is at that
    rate starting from t=0, giving frames at [0, 1/fps, 2/fps, ...]; in
    that mode ``num_frames`` is ignored. If ``target_fps`` is None,
    ``num_frames`` frames are sampled evenly across the full duration.

    Returns a list of HxWx3 RGB numpy arrays.
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cv2 could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if target_fps is not None and target_fps > 0:
        # Sample at target_fps — evenly spaced timestamps
        duration_s = total / fps
        n_target = max(1, int(duration_s * target_fps))
        indices = [int(round(i / target_fps * fps)) for i in range(n_target)]
    else:
        # num_frames evenly spaced across the clip
        if total <= num_frames:
            indices = list(range(total))
        else:
            indices = [int(round(i * (total - 1) / (num_frames - 1)))
                       for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise IOError(f"cv2 failed to read frame {idx} from {video_path}")
        # BGR → RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


class VideoWrapper:
    """Wraps a video-native model for use as a BrainScoreModel activations_model.

    Args:
        model: PyTorch module that accepts a video tensor and returns
            activations. Hook-compatible (has named children layers).
        preprocessing: Callable ``(list_of_frames: List[np.ndarray]) -> torch.Tensor``
            producing a tensor appropriate for the model. Typical layouts:
            ``(T, C, H, W)`` for a single video, or already batched
            ``(1, T, C, H, W)``. The wrapper normalizes to ``(B, T, C, H, W)``.
        identifier: Model identifier for caching.
        target_fps: Frame sampling rate, in Hz. Default 5.
        num_frames: If set, sample exactly this many frames per video
            (overrides target_fps behavior). Defaults to None.
        frame_sampler: Callable used to extract frames. Defaults to
            ``default_uniform_frame_sampler`` (cv2-based).
        forward_kwargs: Extra kwargs passed to the model's forward call
            for every batch (static only — no per-batch dynamic kwargs
            in this version).
        hook_time_axis: Which axis of the hook output corresponds to time.
            Default 1 (after batch). Set to 0 if hook drops the batch dim.
        t_to_time_ms_fn: Callable ``(n_time_steps_out, video_duration_ms) -> list[float]``
            returning the absolute timestamp (ms) for each output time step.
            Default: evenly spread across the video duration.
        batch_size: Number of videos per forward pass. Default 1 — most
            video models have large T×C×H×W tensors.
    """

    def __init__(
        self,
        model,
        preprocessing: Callable[[List[np.ndarray]], 'torch.Tensor'],  # noqa: F821
        identifier: Optional[str] = None,
        target_fps: float = 5.0,
        num_frames: Optional[int] = None,
        frame_sampler: Optional[Callable] = None,
        forward_kwargs: Optional[Dict] = None,
        hook_time_axis: int = 1,
        t_to_time_ms_fn: Optional[Callable[[int, float], List[float]]] = None,
        batch_size: int = 1,
    ):
        import torch
        self._model = model
        self._preprocessing = preprocessing
        self._target_fps = target_fps
        self._num_frames = num_frames
        self._frame_sampler = frame_sampler or default_uniform_frame_sampler
        self._forward_kwargs = dict(forward_kwargs or {})
        self._hook_time_axis = hook_time_axis
        self._t_to_time_ms_fn = t_to_time_ms_fn or _default_time_mapping
        self._batch_size = batch_size

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self._model = self._model.to(self._device)

        self._identifier = identifier or model.__class__.__name__

    @property
    def identifier(self) -> str:
        return self._identifier

    @identifier.setter
    def identifier(self, value: str) -> None:
        self._identifier = value

    def __call__(self, stimuli, layers, stimuli_identifier=None, **kwargs) -> NeuroidAssembly:
        """Extract temporal activations from a batch of videos.

        Args:
            stimuli: StimulusSet with ``video_path`` column (or ``image_file_name``
                for video paths), or list of video file paths.
            layers: List of layer name strings to hook.
            stimuli_identifier: Cache identifier; falls back to
                ``stimuli.identifier`` if available.

        Returns:
            NeuroidAssembly with dims ``(presentation, time_bin, neuroid)``.
            ``presentation`` is one per video (count = len(stimuli)).
            ``time_bin`` is one per output time step (depends on the
            hooked layer's temporal resolution).
        """
        if isinstance(stimuli, StimulusSet):
            return self._from_stimulus_set(stimuli, layers, stimuli_identifier)
        return self._from_paths(list(stimuli), layers, stimuli_identifier)

    def _from_stimulus_set(self, stimulus_set, layers, stimuli_identifier=None):
        if stimuli_identifier is None and hasattr(stimulus_set, 'identifier'):
            stimuli_identifier = stimulus_set.identifier
        paths = self._extract_video_paths(stimulus_set)
        stimulus_ids = list(stimulus_set['stimulus_id'].values)
        activations = self._from_paths_cached(paths, layers, stimuli_identifier)
        activations = self._relabel_stimulus_ids(activations, stimulus_ids)
        activations = self._attach_stimulus_set_meta(activations, stimulus_set)
        return activations

    def _extract_video_paths(self, stimulus_set) -> List[str]:
        for col in ('video_path', 'video_file_name', 'image_file_name', 'filename'):
            if col in stimulus_set.columns:
                return [str(p) for p in stimulus_set[col].values]
        if hasattr(stimulus_set, 'stimulus_paths'):
            return [str(stimulus_set.stimulus_paths[sid])
                    for sid in stimulus_set['stimulus_id']]
        raise ValueError(
            f"Could not find video paths in stimulus set. "
            f"Looked for columns: video_path, video_file_name, "
            f"image_file_name, filename. Got: {list(stimulus_set.columns)}")

    def _from_paths_cached(self, paths, layers, stimuli_identifier=None):
        if self._identifier and stimuli_identifier:
            return self._from_paths_stored(
                identifier=self._identifier,
                stimuli_identifier=stimuli_identifier,
                layers=layers,
                paths=paths,
            )
        return self._from_paths(paths, layers, stimuli_identifier)

    @store_xarray(identifier_ignore=['paths', 'layers'],
                  combine_fields={'layers': 'layer'})
    def _from_paths_stored(self, identifier, layers, stimuli_identifier, paths):
        return self._from_paths(paths, layers, stimuli_identifier)

    def _from_paths(self, paths: List[str], layers: List[str],
                    stimuli_identifier=None) -> NeuroidAssembly:
        if not layers:
            raise ValueError("VideoWrapper requires a layers argument")

        logger.info(f"Running {len(paths)} videos through video model")
        per_video_activations: OrderedDict = OrderedDict()
        per_video_time_ms: List[List[float]] = []

        for batch_start in tqdm(range(0, len(paths), self._batch_size),
                                unit_scale=self._batch_size,
                                desc='video activations'):
            batch_end = min(batch_start + self._batch_size, len(paths))
            batch_paths = paths[batch_start:batch_end]
            batch_outputs, batch_time_ms = self._extract_batch(batch_paths, layers)

            # batch_outputs[layer_name] has shape (B, T_out, features_flat)
            # Append per-video slices.
            if not per_video_activations:
                for layer_name in layers:
                    per_video_activations[layer_name] = []
            for i in range(len(batch_paths)):
                for layer_name in layers:
                    per_video_activations[layer_name].append(
                        batch_outputs[layer_name][i])
                per_video_time_ms.append(batch_time_ms[i])

        return self._package(
            per_video_activations, per_video_time_ms, paths)

    def _extract_batch(self, batch_paths: List[str], layers: List[str]
                       ) -> Tuple[Dict[str, np.ndarray], List[List[float]]]:
        """Run one batch of videos through the model with hooks."""
        import torch

        # Sample and preprocess each video individually, then stack.
        per_video_tensors = []
        per_video_time_ms: List[List[float]] = []
        for path in batch_paths:
            # Sample frames (list of HxWx3 numpy arrays, length T)
            if self._num_frames is not None:
                frames = self._frame_sampler(path, num_frames=self._num_frames)
            else:
                frames = self._frame_sampler(path, num_frames=0,
                                             target_fps=self._target_fps)
            # Record per-frame timestamps (ms) for later remapping
            # to output time steps if hook downsamples.
            import cv2
            cap = cv2.VideoCapture(str(path))
            fps_real = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration_ms = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps_real) * 1000
            cap.release()
            per_video_time_ms.append([0.0])  # placeholder, overridden after hook

            tensor = self._preprocessing(frames)
            if tensor.ndim == 4:  # (T, C, H, W) → add batch dim
                tensor = tensor.unsqueeze(0)
            per_video_tensors.append(tensor)

        # Stack: (B, T, C, H, W) — assumes all videos sampled to same T
        video_batch = torch.cat(per_video_tensors, dim=0).to(self._device)
        # Match model dtype (FP16 models)
        model_dtype = next(self._model.parameters()).dtype
        if video_batch.dtype != model_dtype and video_batch.is_floating_point():
            video_batch = video_batch.to(model_dtype)

        # Register hooks
        layer_outputs: OrderedDict = OrderedDict()
        hooks = []
        for layer_name in layers:
            layer = self._get_layer(layer_name)
            hook = self._register_hook(layer, layer_name, layer_outputs)
            hooks.append(hook)

        self._model.eval()
        with torch.no_grad():
            self._model(video_batch, **self._forward_kwargs)

        for hook in hooks:
            hook.remove()

        # Convert each layer output to (B, T_out, features_flat)
        processed: Dict[str, np.ndarray] = {}
        for layer_name, arr in layer_outputs.items():
            processed[layer_name] = self._flatten_layer_output(arr)

        # Now that we know T_out, compute the per-video time_ms mapping
        # for the first layer (all layers assumed to share time axis size).
        first_layer = layers[0]
        T_out = processed[first_layer].shape[1]
        for i in range(len(batch_paths)):
            per_video_time_ms[i] = self._t_to_time_ms_fn(T_out, duration_ms)

        return processed, per_video_time_ms

    def _flatten_layer_output(self, arr: np.ndarray) -> np.ndarray:
        """Return (B, T, features) from arbitrary hook output.

        Uses self._hook_time_axis to identify the temporal dim in the
        hook's output, then flattens everything after (B, T) into
        ``features``.
        """
        if arr.ndim < 2:
            raise ValueError(
                f"VideoWrapper hook output has ndim={arr.ndim}, expected >=2 "
                f"(at minimum (B, T, ...)).")
        if self._hook_time_axis != 1:
            # Move the time axis into position 1
            arr = np.moveaxis(arr, self._hook_time_axis, 1)
        B, T = arr.shape[0], arr.shape[1]
        return arr.reshape(B, T, -1)

    def _get_layer(self, layer_name: str):
        module = self._model
        for part in layer_name.split('.'):
            module = getattr(module, part, None)
            if module is None:
                raise ValueError(
                    f"Layer '{layer_name}' not found at part '{part}'")
        return module

    def _register_hook(self, layer, layer_name: str, target_dict: OrderedDict):
        def hook_fn(_module, _input, output, name=layer_name):
            if isinstance(output, (tuple, list)):
                output = output[0]
            target_dict[name] = output.detach().cpu().float().numpy()
        return layer.register_forward_hook(hook_fn)

    def _package(
        self,
        per_video_activations: OrderedDict,
        per_video_time_ms: List[List[float]],
        paths: List[str],
    ) -> NeuroidAssembly:
        """Package per-video, per-layer activations into a NeuroidAssembly
        with dims (presentation, time_bin, neuroid)."""
        n_videos = len(paths)
        first_layer = next(iter(per_video_activations))
        # Each per_video_activations[layer] is a list of length n_videos,
        # each entry shape (T_out, features_flat_per_layer).
        T_out = per_video_activations[first_layer][0].shape[0]

        # Assume time axis length is consistent; use the first video's times.
        time_ms = per_video_time_ms[0]

        # Concatenate layers along the feature axis, and stack videos
        layer_tensors = []
        for layer_name, entries in per_video_activations.items():
            # entries: list of (T, F_layer) arrays
            stacked = np.stack(entries, axis=0)  # (n_videos, T, F_layer)
            layer_tensors.append((layer_name, stacked))

        # Concatenate across layers on the feature axis
        feature_arrays = [t for (_, t) in layer_tensors]
        data = np.concatenate(feature_arrays, axis=2)  # (n_videos, T, total_F)

        # Neuroid coords
        neuroid_ids: List[str] = []
        layer_labels: List[str] = []
        for layer_name, stacked in layer_tensors:
            F = stacked.shape[2]
            neuroid_ids.extend([f'{self._identifier}.{layer_name}.{i}'
                                for i in range(F)])
            layer_labels.extend([layer_name] * F)

        # Build stimulus_ids (placeholder — caller may relabel via
        # _relabel_stimulus_ids)
        stimulus_ids = [Path(p).stem for p in paths]

        # Secondary time_bin coord so gather_indexes produces a MultiIndex
        # (single coord → plain Index named after the dim, and the coord
        # name gets lost).
        coords = {
            'stimulus_id': ('presentation', stimulus_ids),
            'video_path': ('presentation', paths),
            'time_bin_center_ms': ('time_bin', time_ms),
            'time_bin_idx': ('time_bin', list(range(len(time_ms)))),
            'neuroid_id': ('neuroid', neuroid_ids),
            'neuroid_num': ('neuroid', list(range(len(neuroid_ids)))),
            'model': ('neuroid', [self._identifier] * len(neuroid_ids)),
            'layer': ('neuroid', layer_labels),
        }

        return NeuroidAssembly(
            data,
            coords=coords,
            dims=['presentation', 'time_bin', 'neuroid'],
        )

    def _relabel_stimulus_ids(
        self,
        assembly: NeuroidAssembly,
        stimulus_ids: List[str],
    ) -> NeuroidAssembly:
        """Replace auto-generated stimulus_ids with canonical ones from the
        caller's StimulusSet. Returns a new assembly (old assembly is
        MultiIndexed and immutable)."""
        # Because gather_indexes MultiIndexes the presentation dim, we
        # need to reset and rebuild. Simpler: construct a fresh assembly
        # with the right stimulus_ids.
        data = assembly.values
        # Collect all coords except the auto-generated stimulus_id
        new_coords = {}
        for coord_name, dims, values in walk_coords(assembly):
            if coord_name == 'stimulus_id':
                continue
            new_coords[coord_name] = (dims, values)
        new_coords['stimulus_id'] = ('presentation', stimulus_ids)
        return NeuroidAssembly(
            data,
            coords=new_coords,
            dims=assembly.dims,
        )

    def _attach_stimulus_set_meta(self, assembly, stimulus_set):
        """Attach remaining stimulus set columns as presentation coords."""
        for column in stimulus_set.columns:
            if column == 'stimulus_id':
                continue
            # Don't overwrite columns we already set (video_path)
            try:
                _ = assembly[column]
                continue
            except (KeyError, AttributeError):
                pass
            try:
                assembly = assembly.assign_coords(
                    {column: ('presentation', list(stimulus_set[column].values))})
            except Exception:
                pass
        return assembly


def _default_time_mapping(n_time_steps_out: int, video_duration_ms: float) -> List[float]:
    """Map output time steps to video time (ms), evenly spread.

    Gives the center of each output window. For ``n_time_steps_out=1``,
    the center is at video_duration_ms / 2.
    """
    if n_time_steps_out == 1:
        return [video_duration_ms / 2.0]
    step_ms = video_duration_ms / n_time_steps_out
    return [step_ms * (i + 0.5) for i in range(n_time_steps_out)]
