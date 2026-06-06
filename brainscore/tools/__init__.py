"""Productionized Brain-Score workflow tools (auto-register, layer mapping)."""
from .layer_mapping import (
    explore_layer_mapping,
    score_approaches,
    sweep_model,
    extract_features_by_layer,
    per_voxel_cv_ridge,
    LayerMappingResult,
)
from .auto_register import (
    inspect_model,
    auto_register,
    scaffold_registration,
    find_block_groups,
    space_layers,
    ModelProfile,
    WrapperRecommendation,
    BlockGroup,
)

__all__ = [
    'explore_layer_mapping', 'score_approaches', 'sweep_model',
    'extract_features_by_layer', 'per_voxel_cv_ridge', 'LayerMappingResult',
    'inspect_model', 'auto_register', 'scaffold_registration',
    'find_block_groups', 'space_layers', 'ModelProfile',
    'WrapperRecommendation', 'BlockGroup',
]
