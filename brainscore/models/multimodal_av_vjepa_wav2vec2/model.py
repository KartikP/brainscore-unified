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


def get_model(identifier: str) -> BrainScoreModel:
    random_video = identifier.startswith('random-vjepa1')
    random_audio = 'random-wav2vec2' in identifier

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
