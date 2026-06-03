"""Visualization tools for the unified interface — model responses mapped onto
the brain, ablation effects, and unit/composite selection populations.

Two tiers:
  * matplotlib-only renderers (parcel heatmaps, network strips, ablation bars,
    selection maps) that always work and are testable offline;
  * nilearn/nibabel cortical-surface renderers (``cortical_surface_map``,
    ``voxel_surface_map``) that produce the inflated-cortex figures in the
    MIRAGE style — run where the surface assets are available.

nilearn/nibabel are imported lazily inside the surface functions, so importing
this package never requires them.
"""
from .brain_map import (
    normalize_values,
    parcel_grid_heatmap,
    network_strip,
    cortical_surface_map,
    voxel_surface_map,
    parcels_to_vertices,
    parcels_to_nifti,
    quickbrain_outline_map,
    fetch_schaefer_fsaverage_annot,
    SCHAEFER_7NETWORKS,
)
from .ablation_plot import (
    ablation_effect_bar,
    response_heatmap,
    before_after_difference,
)
from .unit_selection_plot import (
    composite_selection_map,
    units_per_layer_bar,
    selectivity_histogram,
)
from .scaling_curve import (
    scaling_curve_single,
    scaling_curves_grid,
    normalized_scaling_overlay,
)
from .layer_contribution import layer_modality_heatmap

__all__ = [
    'normalize_values', 'parcel_grid_heatmap', 'network_strip',
    'cortical_surface_map', 'voxel_surface_map', 'fetch_schaefer_fsaverage_annot',
    'parcels_to_vertices', 'parcels_to_nifti', 'quickbrain_outline_map',
    'SCHAEFER_7NETWORKS',
    'ablation_effect_bar', 'response_heatmap', 'before_after_difference',
    'composite_selection_map', 'units_per_layer_bar', 'selectivity_histogram',
    'scaling_curve_single', 'scaling_curves_grid', 'normalized_scaling_overlay',
    'layer_modality_heatmap',
]
