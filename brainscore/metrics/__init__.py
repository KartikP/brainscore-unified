"""Metrics for the unified interface.

Beyond the standard predictivity metrics (re-exported from the domain repos),
this package adds capability-specific metric *types* — currently the
topographic-organization metric (TDANN / TopoLM fold-in), which scores a model's
spatial unit layout against cortical topography rather than its predictivity.
"""
from .topographic import (
    TopographicMetric,
    SelectivityTopographicMetric,
    correlation_distance_profile,
    spatial_smoothness,
    topographic_alignment,
    selectivity_topographic_alignment,
)
from .direct_comparison import DirectComparisonMetric, per_unit_pearson

__all__ = [
    'TopographicMetric', 'SelectivityTopographicMetric',
    'correlation_distance_profile', 'spatial_smoothness',
    'topographic_alignment', 'selectivity_topographic_alignment',
    'DirectComparisonMetric', 'per_unit_pearson',
]

# Register metrics as first-class, loadable plugins: brainscore.load_metric('topographic-alignment').
# Each entry is a factory; any args are forwarded by load_metric. See EXTENDING.md (Seam 3).
from brainscore import metric_registry  # noqa: E402

metric_registry['topographic-alignment'] = lambda n_bins=15: TopographicMetric(n_bins)
# Second topographic axis (selectivity layout) — for Topo-Omni-style models the
# response-correlation axis can't see (see topographic-selectivity-axis-plan).
metric_registry['selectivity-topographic-alignment'] = \
    lambda label_coord='category', top_k_frac=0.1: SelectivityTopographicMetric(label_coord, top_k_frac)
# Direct per-unit comparison for whole-model→whole-brain encoders (e.g. TRIBEv2):
# no fitted readout — correlate predicted vs measured responses per unit.
metric_registry['direct-comparison'] = lambda shuffle_seed=0: DirectComparisonMetric(shuffle_seed)
