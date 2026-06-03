"""Metrics for the unified interface.

Beyond the standard predictivity metrics (re-exported from the domain repos),
this package adds capability-specific metric *types* — currently the
topographic-organization metric (TDANN / TopoLM fold-in), which scores a model's
spatial unit layout against cortical topography rather than its predictivity.
"""
from .topographic import (
    TopographicMetric,
    correlation_distance_profile,
    spatial_smoothness,
    topographic_alignment,
)

__all__ = [
    'TopographicMetric', 'correlation_distance_profile', 'spatial_smoothness',
    'topographic_alignment',
]
