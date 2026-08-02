"""
Brain-Score Unified — single package for multimodal model evaluation.

Provides unified load_model, load_benchmark, and score functions that work
across all modalities. New models and benchmarks that comply with the
Subject/BrainScoreModel interface are registered here. Legacy models
in brainscore_vision and brainscore_language are accessible via fallback
registry lookup.
"""

import logging
from typing import Dict, Any, Callable

from brainscore_core.model_interface import Subject, UnifiedModel, BrainScoreModel
from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Metric, Score

_logger = logging.getLogger(__name__)

model_registry: Dict[str, Callable[[], Subject]] = {}
benchmark_registry: Dict[str, Callable[[], Benchmark]] = {}
metric_registry: Dict[str, Callable[[], Metric]] = {}
data_registry: Dict[str, Callable[..., Any]] = {}
stimulus_set_registry: Dict[str, Callable[..., Any]] = {}


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
        from . import data  # noqa: F401
    except ImportError as e:
        _logger.warning(f"failed to import unified data: {e}")
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


def load_model(identifier: str) -> Subject:
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


def _ensure_legacy_benchmark_modalities(benchmark, modalities) -> Benchmark:
    if not getattr(benchmark, 'required_modalities', None) and not getattr(
            benchmark, 'accepted_modalities', None):
        benchmark.required_modalities = set(modalities)
    return benchmark


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
        benchmark = load_vision_benchmark(identifier)
        return _ensure_legacy_benchmark_modalities(benchmark, {'vision'})
    except (KeyError, ImportError, AssertionError):
        pass

    # Fallback to language
    try:
        from brainscore_language import load_benchmark as load_language_benchmark
        benchmark = load_language_benchmark(identifier)
        return _ensure_legacy_benchmark_modalities(benchmark, {'text'})
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


def load_dataset(identifier: str, *args, **kwargs):
    """Load a dataset (DataAssembly) by identifier.

    Checks unified data plugins first, then falls back to the domain repos so
    ``brainscore.load_dataset`` remains the single entry point alongside
    load_model/load_benchmark/load_metric.
    """
    if identifier in data_registry:
        return data_registry[identifier](*args, **kwargs)

    if args or kwargs:
        raise KeyError(
            f"Dataset '{identifier}' not found in unified registry; "
            "domain fallbacks do not accept loader arguments.")

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
        f"Dataset '{identifier}' not found in unified, vision, or language registries.")


def load_stimulus_set(identifier: str, *args, **kwargs):
    """Load a StimulusSet by identifier (unified first, then domain repos)."""
    if identifier in stimulus_set_registry:
        return stimulus_set_registry[identifier](*args, **kwargs)

    if args or kwargs:
        raise KeyError(
            f"StimulusSet '{identifier}' not found in unified registry; "
            "domain fallbacks do not accept loader arguments.")

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
        f"StimulusSet '{identifier}' not found in unified, vision, or language registries.")


def _resolve_identifier(obj, default: str) -> str:
    """Best-effort identifier for a model/benchmark object (property or method)."""
    ident = getattr(obj, 'identifier', None)
    if callable(ident):
        try:
            ident = ident()
        except Exception:
            ident = None
    return ident or default


def score(model_identifier, benchmark_identifier,
          check_mem: bool = True) -> Score:
    """Score a model on a benchmark.

    Each argument is either an identifier ``str`` — loaded from the unified
    registry (with domain fallbacks) — or an already-constructed object (a
    ``Subject`` / benchmark). Passing objects lets you score a model you just
    built without registering it first::

        model = BrainScoreModel('my-vlm', ...)
        score(model, 'MajajHong2015public.IT-pls-unified')

    then runs the benchmark on the model.
    """
    from brainscore_core.compatibility import (
        check_channel_compatibility,
        check_compatibility,
    )
    from brainscore_core.memory import check_memory
    from brainscore_core.score_metadata import (
        infer_score_protocol,
        requested_output_channels_for_score,
        stamp_score_metadata,
    )

    import time as _time
    # accept either an identifier string or an already-built object
    model = (load_model(model_identifier)
             if isinstance(model_identifier, str) else model_identifier)
    benchmark = (load_benchmark(benchmark_identifier)
                 if isinstance(benchmark_identifier, str) else benchmark_identifier)
    model_id = (model_identifier if isinstance(model_identifier, str)
                else _resolve_identifier(model, 'custom-model'))
    benchmark_id = (benchmark_identifier if isinstance(benchmark_identifier, str)
                    else _resolve_identifier(benchmark, 'custom-benchmark'))
    requested_channels = requested_output_channels_for_score(benchmark)

    check_compatibility(model, benchmark)
    check_channel_compatibility(model, benchmark)
    if check_mem:
        check_memory(model, benchmark)

    _t0 = _time.time()
    result = benchmark(model)
    result.attrs['runtime_sec'] = round(_time.time() - _t0, 2)
    result.attrs['model_identifier'] = model_id
    result.attrs['benchmark_identifier'] = benchmark_id
    stamp_score_metadata(
        result,
        model,
        requested_channels=requested_channels,
        protocol=infer_score_protocol(model, benchmark, requested_channels),
        harness_id='brainscore',
    )
    return result


# Populate unified registries once at import time
_populate_unified_registries()
