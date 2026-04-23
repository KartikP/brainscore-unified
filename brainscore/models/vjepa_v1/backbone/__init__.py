"""Vendored subset of Meta's V-JEPA v1 source for ViT-L/16 loading.

Upstream: https://github.com/facebookresearch/jepa
License: see LICENSE in upstream repo (CC-BY-NC 4.0 for code).

Files vendored (imports rewritten to relative):
    vision_transformer.py  — VisionTransformer + vit_large/huge/giant factories
    patch_embed.py         — PatchEmbed, PatchEmbed3D
    modules.py             — Block, Attention, MLP
    pos_embs.py            — 2D/3D sincos positional embeddings
    tensors.py             — trunc_normal_ helper (from src/utils/)
    masks_utils.py         — apply_masks helper (from src/masks/)

Only the vision_transformer backbone is vendored — not the predictor head,
loss, dataloaders, or training loop. We need the frozen encoder for feature
extraction on Lahner2024.
"""

from .vision_transformer import vit_large, vit_huge, vit_giant, VisionTransformer

__all__ = ['vit_large', 'vit_huge', 'vit_giant', 'VisionTransformer']
