"""Direct per-unit comparison for whole-model to whole-brain encoders (no fitted readout)."""
from brainscore import metric_registry
from .metric import DirectComparisonMetric

__all__ = ['DirectComparisonMetric']

metric_registry['direct-comparison'] = lambda shuffle_seed=0: DirectComparisonMetric(shuffle_seed)
