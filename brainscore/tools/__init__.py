"""Productionized Brain-Score workflow tools (auto-register, layer mapping)."""
from .layer_mapping import (
    explore_layer_mapping,
    score_approaches,
    score_budget_curve,
    effective_dimensionality,
    compute_rdm,
    rsa_score,
    rsa_layer_sweep,
    normalize_by_ceiling,
    sweep_model,
    extract_features_by_layer,
    per_voxel_cv_ridge,
    per_voxel_train_test,
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
from .banded_ridge import ridge_fit_predict, banded_ridge_fit_predict

__all__ = [
    'ridge_fit_predict', 'banded_ridge_fit_predict',
    'explore_layer_mapping', 'score_approaches', 'score_budget_curve',
    'effective_dimensionality', 'compute_rdm', 'rsa_score', 'rsa_layer_sweep',
    'normalize_by_ceiling', 'per_voxel_train_test', 'sweep_model',
    'extract_features_by_layer', 'per_voxel_cv_ridge', 'LayerMappingResult',
    'inspect_model', 'auto_register', 'scaffold_registration',
    'find_block_groups', 'space_layers', 'ModelProfile',
    'WrapperRecommendation', 'BlockGroup',
]
