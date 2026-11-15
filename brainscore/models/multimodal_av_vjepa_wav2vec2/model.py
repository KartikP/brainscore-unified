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


# ── Audio tower config ─────────────────────────────────────────────

# The audio tower is the shared Wav2Vec2-base (``_shared.build_audio_wrapper``).
# Wav2Vec2-base output is (B, T_out, 768) where T_out = floor(samples / 320).
# At 16 kHz that's a 50 Hz step → 20 ms per token. Lahner clips are 3s,
# so each clip yields ~150 audio tokens before chunking; AudioWrapper's
# auto-chunking is harmless here (max_duration_sec defaults to 60s).
AUDIO_LAYER = 'encoder.layers.6'  # mid-encoder block; speech features


# Region → layer mapping spans both towers. Each layer name is in one tower,
# so BrainScoreModel records each region from the tower that has it.
REGION_LAYER_MAP = {
    # Vision tower (V-JEPA v1 ViT-L; mid-encoder block 16 was the
    # per-voxel layer-sweep peak on Lahner2024 — see CLAUDE.md notes).
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
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    from brainscore.models.vjepa_v1.model import (
        _BCTPermuteAdapter, _default_cache_dir, _download_checkpoint,
        _load_vjepa_v1_vitl, _make_preprocessing, _vjepa_v1_post_hook,
        CHECKPOINT_URL, NUM_INPUT_FRAMES,
    )
    from .backbone_inits import randomize_module_in_place

    cache = _default_cache_dir()
    ckpt = _download_checkpoint(CHECKPOINT_URL, cache / 'vitl16.pth.tar', identifier)
    backbone = _load_vjepa_v1_vitl(ckpt).eval()
    if random_init:
        # Random-init the loaded backbone in-place. Deterministic seed.
        randomize_module_in_place(backbone, seed=0)
    model = _BCTPermuteAdapter(backbone).eval()
    preprocessing = _make_preprocessing()
    wrapper = VideoWrapper(
        model=model,
        preprocessing=preprocessing,
        identifier=f'{identifier}-video',
        num_frames=NUM_INPUT_FRAMES,
        post_hook_fn=_vjepa_v1_post_hook,
        batch_size=4,
    )
    return wrapper, model


from ._ids import SUPPORTED_IDENTIFIERS


def get_model(identifier: str) -> BrainScoreModel:
    if identifier not in SUPPORTED_IDENTIFIERS:
        raise AssertionError(
            f"unknown multimodal identifier {identifier!r}; "
            f"expected one of {SUPPORTED_IDENTIFIERS}")
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
