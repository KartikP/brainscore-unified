"""V-JEPA v1 (video) + Wav2Vec2-base (audio): the first registered
multimodal Brain-Score model.

Built specifically to exercise the M12-full unified-interface APIs:
- Two extraction wrappers (`video` + `audio`), one BrainScoreModel
- Each region recorded from the tower whose layers include it: IT from
  the video tower, A1 from the audio tower
- Multi-modality scoring against a multimodal benchmark
  (``Lahner2024-fMRI-naturalistic-multimodal``)

The two backbones are independent — there is NO fusion network. Each
tower extracts features in parallel; the benchmark handles concatenation.
This matches the convention used by TRIBEv2 and similar A/V probes.

Reuses the V-JEPA v1 ViT-L backbone vendored under
``brainscore.models.vjepa_v1.backbone`` (no second download).
"""

import re

import numpy as np

from brainscore import BrainScoreModel
from brainscore.models._shared import build_audio_wrapper


# ── Audio extractor config ─────────────────────────────────────────

# The audio extractor is the shared Wav2Vec2-base (``_shared.build_audio_wrapper``).
# Wav2Vec2-base output is (B, T_out, 768) where T_out = floor(samples / 320).
# At 16 kHz that's a 50 Hz step → 20 ms per token. Lahner clips are 3s,
# so each clip yields ~150 audio tokens before chunking; AudioWrapper's
# auto-chunking is harmless here (max_duration_sec defaults to 60s).
AUDIO_LAYER = 'encoder.layers.6'  # mid-encoder block; speech features


# Region → layer mapping spans both towers. Each layer name is in one tower,
# so BrainScoreModel records each region from the tower that has it.
REGION_LAYER_MAP = {
    # Vision tower (V-JEPA v1 ViT-L; mid-encoder block 16 was the
    # per-voxel layer-sweep peak on Lahner2024).
    'V1': 'backbone.blocks.4',
    'V4': 'backbone.blocks.15',
    'IT': 'backbone.blocks.16',
    # Audio tower (Wav2Vec2-base; mid encoder block).
    'A1': AUDIO_LAYER,
}


# ``vjepa1-wav2vec2-fused``: one region ``cortex`` with 12 bands from the two towers (rule of the study models: at
# most 12): every 4th of the 24 V-JEPA blocks and every 2nd of the 12 Wav2Vec2 encoder layers. A layer path names
# its tower first (``video/`` or ``audio/``).
CORTEX_LAYERS = (tuple(f'video/backbone.blocks.{i}' for i in (3, 7, 11, 15, 19, 23))
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


def _build_video_wrapper(identifier: str, random_init: bool = False):
    """Reuse the V-JEPA v1 backbone + preprocessing from the standalone
    registration. The identifier labels the ``model`` coordinate.

    When ``random_init`` is True, replaces the loaded checkpoint with
    fresh random initialization (deterministic via ``torch.manual_seed(0)``).
    Used for the null-control variants.
    """
    from brainscore.models.vjepa_v1.model import load_backbone, video_extractor
    from .backbone_inits import randomize_module_in_place

    backbone = load_backbone(identifier)
    if random_init:
        # Random-init the loaded backbone in-place. Deterministic seed.
        randomize_module_in_place(backbone, seed=0)
    wrapper, model = video_extractor(backbone, f'{identifier}-video')
    return wrapper, model


def fused_model(identifier: str, video, audio) -> BrainScoreModel:
    """The subject of ``vjepa1-wav2vec2-fused`` from its two towers: ``cortex`` from both, which needs the clip and
    the sound together (``required_channels``)."""
    from brainscore.selection import CompositeSelector
    return BrainScoreModel(
        identifier=identifier,
        region_layer_map={'cortex': CompositeSelector(tuple((layer, None) for layer in CORTEX_LAYERS))},
        extractor={('video', 'audio'): ClipAndSoundExtractor(identifier, {'video': video, 'audio': audio})},
        required_channels=['vision:video', 'audio'],
    )


def get_model(identifier: str) -> BrainScoreModel:
    if identifier == 'vjepa1-wav2vec2-fused':
        return fused_model(identifier, _build_video_wrapper(identifier)[0], build_audio_wrapper(identifier)[0])
    # The identifier names the two towers, each optionally with random weights.
    parts = re.fullmatch(r'(random-)?vjepa1-(random-)?wav2vec2', identifier)
    if parts is None:
        raise ValueError(
            f"{identifier!r} is not a vjepa1+wav2vec2 identifier ([random-]vjepa1-[random-]wav2vec2, "
            f"vjepa1-wav2vec2-fused). "
            f"Next step: load an identifier that brainscore.list_models() lists.")
    random_video, random_audio = bool(parts.group(1)), bool(parts.group(2))

    video_wrapper, video_model = _build_video_wrapper(
        identifier, random_init=random_video)
    audio_wrapper, audio_model = build_audio_wrapper(
        identifier, random_init=random_audio)

    return BrainScoreModel(
        identifier=identifier,
        region_layer_map=REGION_LAYER_MAP,
        extractor={
            'video': video_wrapper,
            'audio': audio_wrapper,
        },
    )
