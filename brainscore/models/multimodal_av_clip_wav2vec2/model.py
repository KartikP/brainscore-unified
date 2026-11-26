"""CLIP ViT-B/32 (vision tower) + Wav2Vec2-base (audio tower).

CLIP is a still-image VLM, so the multimodal benchmark routes through
the frame-aggregation path: extract N frames per clip, run CLIP on
each, mean-pool over frames before regression.

This is the "capable VLM" complement to the V-JEPA-class native-video
combos — it lets us check whether a contrastive image-text encoder
trained on web data carries comparable brain-prediction signal to
representation-reconstruction video models, when both are paired with
the same audio extractor.

Four signal/null permutations registered:
- clip-wav2vec2                  signal vision + signal audio
- random-clip-wav2vec2           null vision   + signal audio
- clip-random-wav2vec2           signal vision + null audio
- random-clip-random-wav2vec2    null vision   + null audio
"""

import re

import numpy as np

from brainscore import BrainScoreModel, LayerSelector
from brainscore.models._shared import build_audio_wrapper, fit_frames


REGION_LAYER_MAP = {
    # Vision tower (CLIP ViT-B/32 vision_model — paths relative to
    # the wrapped sub-module). Per the standalone clip-vit-b-32 entry.
    'V1': 'encoder.layers.1',
    'V2': 'encoder.layers.3',
    'V4': 'encoder.layers.6',
    'IT': 'encoder.layers.10',
    # Audio tower
    'A1': 'encoder.layers.6',
}

REGION_MODALITY_MAP = {
    'V1': 'vision',
    'V2': 'vision',
    'V4': 'vision',
    'IT': 'vision',
    'A1': 'audio',
}


# ``clip-wav2vec2-fused``: one region ``cortex`` with 12 bands from the two towers (at most 12): every 2nd of the
# 12 CLIP vision blocks and every 2nd of the 12 Wav2Vec2 encoder layers. A layer path names its tower first.
# CLIP sees FRAMES frames evenly spaced over the clip, one at a time; its features are averaged over the frames.
FRAMES = 8
CORTEX_LAYERS = (tuple(f'video/model.encoder.layers.{i}' for i in (1, 3, 5, 7, 9, 11))
                 + tuple(f'audio/encoder.layers.{i}' for i in (1, 3, 5, 7, 9, 11)))


class ClipAndSoundExtractor:
    """One extractor for the clip (``vision:video``) and the sound (``audio``) of each row: both towers.

    A ``BrainScoreModel`` region takes its layers from one extractor, so a region made of video blocks and audio
    layers needs one extractor that runs both towers. A layer path is ``<tower>/<layer of that tower>``. Each tower
    runs on its own column of the table; its response is averaged over its time bins (the per-row form of a
    benchmark that takes one value per row, as ``Paugam2025-fMRI-mario-cortex`` does), and the towers' layers are
    joined on ``neuroid`` in the order asked.
    Limit: the time steps inside a row are averaged, so a benchmark that reads the time course inside a row gets
    one step. The audio tower pads the sounds of one call to the longest; a shorter sound's mean includes the
    padding.
    The same class is in ``multimodal_av_vjepa_wav2vec2``: one small copy in each model folder.
    Limit: this extractor exists because a region cannot yet take its layers from several extractors; it goes when
    that is possible.
    """

    def __init__(self, identifier: str, towers: dict) -> None:
        self.identifier = identifier
        self._towers = dict(towers)

    def layer_names(self) -> list:
        return [f'{tower}/{layer}' for tower, extractor in self._towers.items() for layer in extractor.layer_names()]

    def __call__(self, stimuli, layers):
        from brainscore_core.assembly_builder import make_assembly, make_layer_neuroid_coords
        layers = list(layers)
        unknown = [layer for layer in layers if layer.partition('/')[0] not in self._towers]
        if unknown:
            raise ValueError(f'{unknown} name no tower; a layer path is <tower>/<layer>, with a tower of '
                             f'{sorted(self._towers)}.')
        rows = [str(i) for i in stimuli['stimulus_id']]
        per_layer = {}
        for tower, extractor in self._towers.items():
            asked = [layer.partition('/')[2] for layer in layers if layer.partition('/')[0] == tower]
            if not asked:
                continue
            response = extractor(stimuli, layers=asked)
            steps = [dim for dim in response.dims if dim not in ('presentation', 'neuroid')]
            if steps not in ([], ['time_bin']):
                raise ValueError(f'the {tower} tower answered dims {response.dims}; a tower answers presentation, '
                                 f'neuroid and at most time_bin.')
            if [str(i) for i in response['stimulus_id'].values] != rows:
                raise ValueError(f'the {tower} tower did not answer the rows of the table in their order.')
            values = np.asarray(response.transpose('presentation', *steps, 'neuroid').values)
            if steps:
                values = values.mean(axis=1)
            labels = np.asarray(response['layer'].values).astype(str)
            for layer in asked:
                per_layer[f'{tower}/{layer}'] = values[:, labels == layer]
        neuroid = {}
        for layer in layers:
            for name, (_, values) in make_layer_neuroid_coords(self.identifier, layer,
                                                                per_layer[layer].shape[1]).items():
                neuroid.setdefault(name, []).extend(values)
        values = np.concatenate([per_layer[layer] for layer in layers], axis=1)
        presentation = {column: ('presentation', list(stimuli[column])) for column in stimuli.columns}
        return make_assembly(values, coords={**presentation,
                                             **{name: ('neuroid', v) for name, v in neuroid.items()}},
                             dims=['presentation', 'neuroid'])


def frame_mean(output: np.ndarray) -> np.ndarray:
    """A layer's output for ``B * FRAMES`` frames ``(B * FRAMES, tokens, H)`` as ``(B, 1, tokens * H)``: the mean
    over the frames of each clip, taken here so a chunk of clips keeps one step per clip (the extractor averages
    the steps anyway)."""
    return output.reshape(output.shape[0] // FRAMES, FRAMES, -1).mean(axis=1, keepdims=True)


def _frames_extractor(identifier: str, random_init: bool = False):
    """CLIP's vision model on ``FRAMES`` frames of each clip, one frame at a time (a ``VideoWrapper``); a layer's
    output of each frame is all its tokens, as the picture extractor of ``clip-vit-b-32`` records them."""
    import torch
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    from brainscore.models.clip_vit_b_32.model import load_clip

    class Frames(torch.nn.Module):
        """The vision model on each frame of a clip batch ``(B, FRAMES, 3, H, W)``."""

        def __init__(self, vision_model):
            super().__init__()
            self.model = vision_model

        def forward(self, video):
            return self.model(pixel_values=video.flatten(0, 1))

    clip_model, processor = load_clip(random_init=random_init)

    def preprocess(frames):
        return processor(images=fit_frames(frames, FRAMES), return_tensors='pt')['pixel_values']

    return VideoWrapper(model=Frames(clip_model.vision_model).eval(), preprocessing=preprocess,
                        identifier=f'{identifier}-video', target_fps=None, num_frames=FRAMES,
                        post_hook_fn=frame_mean,
                        batch_size=8)


def fused_model(identifier: str, video, audio) -> BrainScoreModel:
    """The subject of ``clip-wav2vec2-fused`` from its two towers: ``cortex`` from both, which needs the clip and
    the sound together (``required_channels``)."""
    from brainscore.selection import CompositeSelector
    return BrainScoreModel(
        identifier=identifier,
        region_layer_map={'cortex': CompositeSelector(tuple((layer, None) for layer in CORTEX_LAYERS))},
        extractor={('video', 'audio'): ClipAndSoundExtractor(identifier, {'video': video, 'audio': audio})},
        required_channels=['vision:video', 'audio'],
    )


def _build_vision_wrapper(combo_identifier: str, random_init: bool = False):
    """The picture extractor of ``clip-vit-b-32`` (or of ``random-vit-b-32`` with ``random_init``)."""
    from brainscore.models.clip_vit_b_32.model import load_clip, picture_extractor
    clip_model, processor = load_clip(random_init=random_init)
    # The activations carry the backbone's name; the cache key is the combo's.
    vision_wrapper = picture_extractor(clip_model, processor,
                                   'random-clip-vit-b-32' if random_init else 'clip-vit-b-32',
                                   backbone_id=f'{combo_identifier}-vision')
    return vision_wrapper


def get_model(identifier: str) -> BrainScoreModel:
    if identifier == 'clip-wav2vec2-fused':
        return fused_model(identifier, _frames_extractor(identifier), build_audio_wrapper(identifier)[0])
    # The identifier names the two towers, each optionally with random weights.
    parts = re.fullmatch(r'(random-)?clip-(random-)?wav2vec2', identifier)
    if parts is None:
        raise ValueError(
            f"{identifier!r} is not a clip+wav2vec2 identifier ([random-]clip-[random-]wav2vec2, clip-wav2vec2-fused). "
            f"Next step: load an identifier that brainscore.list_models() lists.")
    random_vision, random_audio = bool(parts.group(1)), bool(parts.group(2))

    vision_wrapper = _build_vision_wrapper(
        identifier, random_init=random_vision)
    audio_wrapper, _ = build_audio_wrapper(
        identifier, random_init=random_audio)

    return BrainScoreModel(
        identifier=identifier,
        region_layer_map={
            region: LayerSelector(name=layer, modality=REGION_MODALITY_MAP.get(region))
            for region, layer in REGION_LAYER_MAP.items()
        },
        extractor={
            'vision': vision_wrapper,
            'audio': audio_wrapper,
        },
    )
