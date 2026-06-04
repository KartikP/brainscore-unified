"""Induced-dyslexia perturbation benchmark — registered variants.

A causal ablation benchmark: localize word-form units, switch them off, measure
the reading deficit against a matched random-ablation control. Built on the core
``UnitSelection`` family (``FunctionalSelection`` / ``RandomSelection`` /
``CompositeSelector``) + ``StateChange``. See ``benchmark.py``.

Two pinned variants by presentation, mirroring the ROAR reading test it reuses:
- ``Yeatman2021-induced_dyslexia-image`` — word images
- ``Yeatman2021-induced_dyslexia-text``  — word strings
"""
from brainscore import benchmark_registry
from .benchmark import InducedDyslexia

benchmark_registry['Yeatman2021-induced_dyslexia-image'] = (
    lambda: InducedDyslexia(modality='vision'))
benchmark_registry['Yeatman2021-induced_dyslexia-text'] = (
    lambda: InducedDyslexia(modality='text'))
