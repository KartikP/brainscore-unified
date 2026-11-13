"""Productionized Brain-Score workflow tools (auto-register, layer mapping).

The tool registry and auto-register (standard library only) load with the
package. Layer mapping and ``attach_tools`` load at the first use of one of
their names. Auto-register stays eager because its function has the name of
its module: a lazy load would let an earlier ``import
brainscore.tools.auto_register`` put the module where the function belongs.
"""
from importlib import import_module

from ._registry import list_tools, load_tool, register_tool
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

_LAYER_MAPPING = (
    'explore_layer_mapping', 'score_approaches', 'score_budget_curve',
    'effective_dimensionality', 'compute_rdm', 'rsa_score', 'rsa_layer_sweep',
    'normalize_by_ceiling', 'sweep_model', 'extract_features_by_layer',
    'per_voxel_cv_ridge', 'per_voxel_train_test', 'LayerMappingResult',
)
# Each lazily loaded name and the module that defines it.
_MODULE_OF = {**dict.fromkeys(_LAYER_MAPPING, '.layer_mapping'), 'attach_tools': '._scope'}


def __getattr__(name):
    if name not in _MODULE_OF:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(_MODULE_OF[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_MODULE_OF))


__all__ = [
    'attach_tools', 'list_tools', 'load_tool', 'register_tool', *_LAYER_MAPPING,
    'inspect_model', 'auto_register', 'scaffold_registration',
    'find_block_groups', 'space_layers', 'ModelProfile',
    'WrapperRecommendation', 'BlockGroup',
]
