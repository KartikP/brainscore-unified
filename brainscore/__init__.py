"""
Brain-Score Unified — single package for multimodal model evaluation.

Provides unified load_model, load_benchmark, and score functions that work
across all modalities. New models and benchmarks that comply with the
UnifiedModel/BrainScoreModel interface are registered here. Legacy models
in brainscore_vision and brainscore_language are accessible via fallback
registry lookup.
"""

import logging
from typing import Dict, Any, Callable

from brainscore_core.model_interface import UnifiedModel, BrainScoreModel
from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Metric, Score

_logger = logging.getLogger(__name__)

model_registry: Dict[str, Callable[[], UnifiedModel]] = {}
benchmark_registry: Dict[str, Callable[[], Benchmark]] = {}
metric_registry: Dict[str, Callable[[], Metric]] = {}


def _populate_unified_registries() -> None:
    """Import benchmark + model subpackages so they register factories.

    Kept internal and called once at import time. Imports are cheap (each
    subpackage just registers a factory; heavy data loading happens at
    factory call time).
    """
    try:
        from . import benchmarks  # noqa: F401
    except ImportError as e:
        _logger.warning(f"failed to import unified benchmarks: {e}")
    try:
        from . import models  # noqa: F401
    except ImportError as e:
        _logger.warning(f"failed to import unified models: {e}")


def load_model(identifier: str) -> UnifiedModel:
    """Load a model by identifier.

    Checks the unified registry first, then falls back to domain-specific
    registries (brainscore_vision, brainscore_language) for legacy models.
    Legacy models are auto-wrapped in their domain adapter.
    """
    # Check unified registry first
    if identifier in model_registry:
        return model_registry[identifier]()

    # Fallback to vision
    try:
        from brainscore_vision import load_model as load_vision_model
        return load_vision_model(identifier)
    except (KeyError, ImportError, AssertionError):
        pass

    # Fallback to language
    try:
        from brainscore_language import load_model as load_language_model
        return load_language_model(identifier)
    except (KeyError, ImportError, AssertionError):
        pass

    raise KeyError(
        f"Model '{identifier}' not found in unified, vision, or language registries."
    )


def load_benchmark(identifier: str) -> Benchmark:
    """Load a benchmark by identifier.

    Checks the unified registry first, then falls back to domain-specific
    registries.
    """
    if identifier in benchmark_registry:
        return benchmark_registry[identifier]()

    # Fallback to vision
    try:
        from brainscore_vision import load_benchmark as load_vision_benchmark
        return load_vision_benchmark(identifier)
    except (KeyError, ImportError, AssertionError):
        pass

    # Fallback to language
    try:
        from brainscore_language import load_benchmark as load_language_benchmark
        return load_language_benchmark(identifier)
    except (KeyError, ImportError, AssertionError):
        pass

    raise KeyError(
        f"Benchmark '{identifier}' not found in unified, vision, or language registries."
    )


def score(model_identifier: str, benchmark_identifier: str) -> Score:
    """Score a model on a benchmark.

    Loads both from the unified registry (with domain fallbacks),
    then runs the benchmark on the model.
    """
    model = load_model(model_identifier)
    benchmark = load_benchmark(benchmark_identifier)
    result = benchmark(model)
    result.attrs['model_identifier'] = model_identifier
    result.attrs['benchmark_identifier'] = benchmark_identifier
    return result


# Populate unified registries once at import time
_populate_unified_registries()
