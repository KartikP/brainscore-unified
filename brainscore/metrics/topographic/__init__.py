"""Topographic-alignment metrics: a model's spatial unit layout against cortical topography."""
from brainscore import metric_registry
from .metric import (
    TopographicMetric,
    SelectivityTopographicMetric,
    correlation_distance_profile,
    spatial_smoothness,
    topographic_alignment,
    selectivity_topographic_alignment,
    selectivity_maps,           # importable from this path as before; not in __all__
    selectivity_centroids,
)

__all__ = [
    'TopographicMetric', 'SelectivityTopographicMetric',
    'correlation_distance_profile', 'spatial_smoothness',
    'topographic_alignment', 'selectivity_topographic_alignment',
]

metric_registry['topographic-alignment'] = lambda n_bins=15: TopographicMetric(n_bins)
# Second axis (selectivity layout), for models the response-correlation axis cannot see.
metric_registry['selectivity-topographic-alignment'] = \
    lambda label_coord='category', top_k_frac=0.1: SelectivityTopographicMetric(label_coord, top_k_frac)
