"""Direct per-unit comparison for whole-model to whole-brain encoders (no fitted readout)."""
from brainscore import metric_registry
from .metric import DirectComparisonMetric, per_unit_pearson

__all__ = ['DirectComparisonMetric', 'per_unit_pearson']

metric_registry['direct-comparison'] = lambda shuffle_seed=0: DirectComparisonMetric(shuffle_seed)
