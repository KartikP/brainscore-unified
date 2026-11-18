"""Registers your metric in brainscore.metric_registry so load_metric('your-metric') works.

To install: copy this folder to ``brainscore/metrics/your_name/``; it registers by
its presence (no other file changes). Put the
test in ``tests/test_metrics/test_<your_name>.py``.
"""
from brainscore import metric_registry
from .metric import YourMetric

# The value is a factory; any args you pass to load_metric('your-metric', ...) are forwarded.
metric_registry['your-metric'] = lambda **params: YourMetric(**params)
