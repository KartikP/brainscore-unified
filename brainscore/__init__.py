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

    Kept internal and called once at import time. Imports are cheap by
    contract: each plugin __init__ registers a factory that defers
    ``from .model import get_model`` to call time, so heavy deps
    (torch/transformers/cv2/sklearn) load only when a model is actually
    loaded — not on ``import brainscore``. Guarded by
    ``tests/test_import_hygiene.py``.
    """
    try:
        from . import benchmarks  # noqa: F401
    except ImportError as e:
        _logger.warning(f"failed to import unified benchmarks: {e}")
    try:
        from . import models  # noqa: F401
    except ImportError as e:
        _logger.warning(f"failed to import unified models: {e}")
    try:
        from . import metrics  # noqa: F401
    except ImportError as e:
        _logger.warning(f"failed to import unified metrics: {e}")


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


def load_metric(identifier: str, *args, **kwargs) -> Metric:
    """Load a metric by identifier.

    Checks the unified registry first, then falls back to the brainscore_vision
    registry. Metric factories may take arguments (e.g. a region or n_components),
    forwarded here.
    """
    if identifier in metric_registry:
        return metric_registry[identifier](*args, **kwargs)

    try:
        from brainscore_vision import load_metric as load_vision_metric
        return load_vision_metric(identifier, *args, **kwargs)
    except (KeyError, ImportError, AssertionError):
        pass

    raise KeyError(
        f"Metric '{identifier}' not found in unified or vision registries."
    )


def load_dataset(identifier: str):
    """Load a dataset (DataAssembly) by identifier.

    Datasets live in the domain repos, not the unified package — this is a
    convenience passthrough so ``brainscore.load_dataset`` is the single entry
    point alongside load_model/load_benchmark/load_metric. Tries vision then
    language. The heavy import is deferred to call time (import hygiene).
    """
    try:
        from brainscore_vision import load_dataset as _vision
        return _vision(identifier)
    except (KeyError, ImportError, AssertionError):
        pass
    try:
        from brainscore_language import load_dataset as _language
        return _language(identifier)
    except (KeyError, ImportError, AssertionError):
        pass
    raise KeyError(
        f"Dataset '{identifier}' not found in vision or language registries.")


def load_stimulus_set(identifier: str):
    """Load a StimulusSet by identifier (domain-repo passthrough; see load_dataset)."""
    try:
        from brainscore_vision import load_stimulus_set as _vision
        return _vision(identifier)
    except (KeyError, ImportError, AssertionError):
        pass
    try:
        from brainscore_language import load_stimulus_set as _language
        return _language(identifier)
    except (KeyError, ImportError, AssertionError):
        pass
    raise KeyError(
        f"StimulusSet '{identifier}' not found in vision or language registries.")


def score(model_identifier: str, benchmark_identifier: str,
          check_mem: bool = True) -> Score:
    """Score a model on a benchmark.

    Loads both from the unified registry (with domain fallbacks),
    then runs the benchmark on the model.
    """
    from brainscore_core.compatibility import check_compatibility
    from brainscore_core.memory import check_memory

    import time as _time
    model = load_model(model_identifier)
    benchmark = load_benchmark(benchmark_identifier)

    check_compatibility(model, benchmark)
    if check_mem:
        check_memory(model, benchmark)

    _t0 = _time.time()
    result = benchmark(model)
    result.attrs['runtime_sec'] = round(_time.time() - _t0, 2)
    result.attrs['model_identifier'] = model_identifier
    result.attrs['benchmark_identifier'] = benchmark_identifier
    return result


# Populate unified registries once at import time
_populate_unified_registries()
