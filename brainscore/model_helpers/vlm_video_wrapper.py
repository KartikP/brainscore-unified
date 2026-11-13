"""Clips through a vision-language model's own video processor (Qwen-VL style).

Register it under the ``video`` key next to the still-image extractor, so the
subject declares that it takes clips natively (``vision:video``):

    extractor={'vision': VLMVisionWrapper(model.model.visual, processor, ...),
               'video': VLMVideoWrapper(model.model.visual, processor, identifier=...)}

Every frame of the clip is decoded and handed to the processor with the clip's
real frame rate; the processor samples at its own rate (Qwen: 2 fps) and merges
frame pairs into temporal patches. Each temporal patch becomes one time bin,
spanning ``temporal_patch_size / fps`` seconds, with activity averaged over
the patch's spatial positions. Output: ``(presentation, time_bin, neuroid)``.
"""
from typing import List

import numpy as np

from brainscore_core.responses import part_columns
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore.model_helpers._extractors import LiveRecording, as_float32, value


def read_clip(path):
    """All frames of a video as ``(T, H, W, 3)`` uint8 RGB, and its frame rate."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f'Cannot open video {path}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames or not fps:
        raise ValueError(f'No frames or frame rate in {path}')
    return np.stack(frames), float(fps)


def clip_metadata(frames, fps):
    try:
        from transformers.video_utils import VideoMetadata
        return VideoMetadata(total_num_frames=len(frames), fps=fps, duration=len(frames) / fps)
    except ImportError:          # older transformers take a plain mapping
        return {'total_num_frames': len(frames), 'fps': fps, 'duration': len(frames) / fps}


def _grid(args, kwargs):
    """The ``grid_thw`` a pass was given, as values: one row, one clip."""
    grid = kwargs.get('grid_thw', args[1] if len(args) > 1 else None)
    if grid is None or len(grid) != 1:
        raise ValueError('A clip is recorded from a pass given one grid_thw row (one clip); this '
                         f'pass gives {None if grid is None else len(grid)}.')
    return value(grid)


class VLMVideoWrapper:
    def __init__(self, model, processor, identifier: str, *, backbone_id: str = None):
        self._model, self._processor = model, processor
        self._identifier, self._backbone_id = identifier, backbone_id or identifier
        video = processor.video_processor
        self._step_ms = 1000.0 * video.temporal_patch_size / video.fps

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def backbone_id(self) -> str:
        return self._backbone_id

    def __call__(self, stimuli, layers: List[str], stimuli_identifier=None, **kwargs):
        import torch
        paths = self._paths(stimuli)
        per_clip = []
        for path in paths:
            inputs = self._inputs(path)
            with LiveRecording(self._model, self._model, layers, as_float32, self._steps,
                               _grid) as recording, torch.no_grad():
                parameter = next(self._model.parameters())
                self._model(inputs['pixel_values_videos'].to(parameter.device, parameter.dtype),
                            grid_thw=inputs['video_grid_thw'].to(parameter.device))
                per_clip.append(recording.take())
        steps = {len(clip[layers[0]]) for clip in per_clip}
        if len(steps) != 1:
            raise ValueError(f'Clips gave different numbers of time steps {sorted(steps)}; '
                             'present clips of one length together.')
        return self._package({layer: np.stack([clip[layer] for clip in per_clip]) for layer in layers},
                             {column: list(stimuli[column].values) for column in stimuli.columns})

    def record(self, layers: List[str], stimuli=None):
        """Record ``layers`` while another caller (e.g. generation) runs the model on a clip.

        ``take()`` gives ``{layer: assembly}`` for the first pass since the previous
        take, time-resolved by the functions of ``__call__`` with the ``grid_thw`` of
        that pass. Clips of ``stimuli`` with different numbers of time steps are
        refused before any pass; a pass of several clips, and a later pass of the
        presentation with other input, before they run.
        """
        if stimuli is not None and len({int(self._inputs(path)['video_grid_thw'][0][0])
                                        for path in self._paths(stimuli)}) > 1:
            raise ValueError('The clips give different numbers of time steps, so they share no time '
                             'axis; present clips of one length together.')
        return LiveRecording(self._model, self._model, layers, as_float32, lambda activity, grid, row: {
            layer: self._package({layer: steps[None]}, {'stimulus_id': ['?']})
            for layer, steps in self._steps(activity, grid, row).items()}, _grid, one_input=True)

    def _inputs(self, path):
        """The processor inputs of a clip: every frame, with the clip's real frame rate."""
        frames, fps = read_clip(path)
        return self._processor.video_processor(
            videos=[frames], video_metadata=[clip_metadata(frames, fps)], return_tensors='pt')

    def _steps(self, activity, grid, _row=None):
        """{layer: (time steps, features)} for one clip: each step's positions averaged."""
        return {layer: self._per_time(a, int(grid[0][0])) for layer, a in activity.items()}

    @staticmethod
    def _per_time(activity, t):
        patches = activity.reshape(-1, activity.shape[-1])
        if patches.shape[0] % t:
            raise ValueError(f'{patches.shape[0]} positions do not divide into {t} time steps')
        return patches.reshape(t, -1, patches.shape[-1]).mean(axis=1)

    def _package(self, activity, presentation):
        layers = list(activity)
        values = np.concatenate([activity[layer] for layer in layers], axis=2)
        steps = values.shape[1]
        neuroids = [(layer, unit) for layer in layers for unit in range(activity[layer].shape[2])]
        return NeuroidAssembly(values, dims=['presentation', 'time_bin', 'neuroid'], coords={
            **{name: ('presentation', values_) for name, values_ in presentation.items()},
            'time_bin_start_ms': ('time_bin', [k * self._step_ms for k in range(steps)]),
            'time_bin_end_ms': ('time_bin', [(k + 1) * self._step_ms for k in range(steps)]),
            'neuroid_id': ('neuroid', [f'{self._identifier}.{layer}.{unit}' for layer, unit in neuroids]),
            'layer': ('neuroid', [layer for layer, _ in neuroids])})

    def layer_names(self):
        """Layer paths this extractor can record, relative to the module it wraps."""
        return [name for name, _ in self._model.named_modules(remove_duplicate=False) if name]

    @staticmethod
    def _paths(stimuli):
        clip_columns = part_columns('video')
        for column in clip_columns:
            if column in stimuli.columns:
                return [str(p) for p in stimuli[column].values]
        paths = getattr(stimuli, 'stimulus_paths', None)
        if paths:
            return [str(paths[sid]) for sid in stimuli['stimulus_id']]
        raise ValueError(f'No clip column {clip_columns} in {list(stimuli.columns)}')


# Recording several layers in one call gives, for each layer, exactly what a
# call for that layer alone gives, so the regions of one session share one call
# (brainscore._model.channels). Set outside the class body: the class source is
# part of the activation cache key, and this line must not change that key.
VLMVideoWrapper.independent_layers = True
