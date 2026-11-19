"""Induced-dyslexia perturbation benchmark — registered variants.

A causal ablation benchmark: localize word-form units, switch them off, measure
the reading deficit against a matched random-ablation control. Built on the core
``UnitSelection`` family (``FunctionalSelection`` / ``RandomSelection`` /
``CompositeSelector``) + ``StateChange``. See ``benchmark.py``.

Two pinned variants by presentation, mirroring the ROAR reading test it reuses:
- ``Yeatman2021-induced_dyslexia-image`` — word images
- ``Yeatman2021-induced_dyslexia-text``  — word strings
"""
from functools import partial

from brainscore import benchmark_registry


def _load(**configuration):
    from .benchmark import InducedDyslexia
    return InducedDyslexia(**configuration)


benchmark_registry['Yeatman2021-induced_dyslexia-image'] = partial(
    _load, channels=('vision:image',), identifier='Yeatman2021-induced_dyslexia-image')
benchmark_registry['Yeatman2021-induced_dyslexia-text'] = partial(
    _load, channels=('text',), identifier='Yeatman2021-induced_dyslexia-text')
