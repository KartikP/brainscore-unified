"""
Brain-Score Unified — single package for multimodal model evaluation.

Provides unified load_model, load_benchmark, and score functions that work
across all modalities. New models and benchmarks that comply with the
Subject/BrainScoreModel interface are registered here. Legacy models
in brainscore_vision and brainscore_language are accessible via fallback
registry lookup.
"""

import logging
from pathlib import Path
from brainscore_core.plugin_management.import_plugin import PluginNotFoundError
from typing import Dict, Any, Callable

from brainscore_core.model_interface import Subject, BrainScoreModel
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


class LoadError(RuntimeError):
    """A registered model or benchmark that cannot load here, with the reason and a next step."""


_ISSUES = {'vision': 'https://github.com/brain-score/vision/issues',
           'language': 'https://github.com/brain-score/language/issues'}
_GONE_STATUS = {403, 404, 410}
_GONE_CODES = {'403', '404', 'NoSuchKey', 'NoSuchBucket', 'AccessDenied'}
_GONE_CLASSES = {'RepositoryNotFoundError', 'EntryNotFoundError', 'RevisionNotFoundError',
                 'GatedRepoError', 'FileURLRetrievalError'}


def _gone(error) -> bool:
    """Whether an error (or one it was raised from) says a download no longer exists."""
    while error is not None:
        status = getattr(error, 'code', None) or getattr(getattr(error, 'response', None), 'status_code', None)
        aws = getattr(error, 'response', None)
        aws_code = aws.get('Error', {}).get('Code') if isinstance(aws, dict) else None
        if (type(error).__name__ in _GONE_CLASSES or status in _GONE_STATUS
                or aws_code in _GONE_CODES):
            return True
        error = error.__cause__ or error.__context__
    return False


class _explained:
    """Turn a known load failure into a LoadError with a readable reason and next step."""

    def __init__(self, identifier, domain, files):
        self.identifier, self.domain, self.files = identifier, domain, files

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is None or isinstance(error, (PluginNotFoundError, LoadError)):
            return False
        from brainscore_core.plugin_management.import_plugin import PluginDependencyError
        if isinstance(error, PluginDependencyError):
            raise LoadError(str(error)) from error
        if _gone(error):
            raise LoadError(
                f"'{self.identifier}' can't load: its {self.files} are no longer at the address "
                f"recorded for it ({type(error).__name__}: {str(error).splitlines()[0][:200]}). "
                f"This is a problem with the {self.domain} plugin, not your setup. "
                f"Please tell a maintainer, quoting this message: {_ISSUES[self.domain]}") from error
        if isinstance(error, ModuleNotFoundError):
            raise LoadError(
                f"'{self.identifier}' needs the package '{error.name}', which is not installed. "
                f"Next step: load it with BS_INSTALL_DEPENDENCIES unset (or 'yes'), so Brain-Score "
                f"installs what the plugin lists, or install '{error.name}' yourself.") from error
        return False


def _registered(domain: str, folder: str, registry: str):
    """Identifiers registered by a domain library's plugins, read from their
    ``__init__.py`` files without importing them (as the loader finds them)."""
    import importlib.util
    import re
    spec = importlib.util.find_spec(f'brainscore_{domain}')
    if spec is None or not spec.submodule_search_locations:
        return []
    root = Path(list(spec.submodule_search_locations)[0]) / folder
    pattern = re.compile(registry + r"""\[['"]([^'"]+)['"]\]""")       # either quote, as the loader accepts
    found = []
    for init in sorted(root.glob('*/__init__.py')):
        if not init.parent.name.startswith(('_', '.')):
            found += pattern.findall(init.read_text(encoding='utf-8', errors='replace'))
    return found


def _listing(unified, folder, registry):
    import pandas as pd
    rows = [(identifier, 'unified') for identifier in sorted(unified)]
    for domain in ('vision', 'language'):
        rows += [(identifier, domain) for identifier in _registered(domain, folder, registry)]
    table = pd.DataFrame(rows, columns=['identifier', 'source'])
    return table.drop_duplicates('identifier').reset_index(drop=True)    # load order: unified first


def list_models():
    """Every model ``load_model`` can find, as a table of ``identifier`` and ``source``
    (unified, vision or language). Nothing is downloaded or imported to list them."""
    return _listing(model_registry, 'models', 'model_registry')


def list_benchmarks():
    """Every benchmark ``load_benchmark`` can find, as a table of ``identifier`` and ``source``."""
    return _listing(benchmark_registry, 'benchmarks', 'benchmark_registry')


def load_model(identifier: str) -> Subject:
    """Load a model by identifier.

    Checks the unified registry first, then falls back to domain-specific
    registries (brainscore_vision, brainscore_language) for legacy models.
    Legacy models are auto-wrapped in their domain adapter. A model that cannot
    load for a known reason (its weights are gone, its packages conflict with
    this environment) raises ``LoadError`` saying why and what to try next.
    """
    if identifier in model_registry:
        return model_registry[identifier]()
    for domain in ('vision', 'language'):
        try:
            with _explained(identifier, domain, 'weights'):
                return __import__(f'brainscore_{domain}', fromlist=['load_model']).load_model(identifier)
        except PluginNotFoundError:
            continue
    raise KeyError(
        f"Model '{identifier}' not found in unified, vision, or language registries; "
        f"brainscore.list_models() lists them all."
    )


def load_benchmark(identifier: str, **configuration: Any) -> Benchmark:
    """Load a benchmark by identifier.

    Checks the unified registry first, then falls back to domain-specific
    registries. Configuration keywords are passed to unified benchmark factories.
    Domain fallbacks accept identifiers only.
    """
    from brainscore_core.compatibility import ensure_legacy_benchmark_modalities

    if identifier in benchmark_registry:
        return benchmark_registry[identifier](**configuration)

    if configuration:
        raise TypeError('Benchmark configuration is supported only by unified registry factories')

    for domain, modality in (('vision', 'vision'), ('language', 'text')):
        try:
            with _explained(identifier, domain, 'data'):
                benchmark = __import__(f'brainscore_{domain}', fromlist=['load_benchmark']).load_benchmark(identifier)
            return ensure_legacy_benchmark_modalities(benchmark, {modality})
        except PluginNotFoundError:
            continue
    raise KeyError(
        f"Benchmark '{identifier}' not found in unified, vision, or language registries; "
        f"brainscore.list_benchmarks() lists them all."
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
    except PluginNotFoundError:
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
    except PluginNotFoundError:
        pass
    try:
        from brainscore_language import load_dataset as _language
        return _language(identifier)
    except PluginNotFoundError:
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
    except PluginNotFoundError:
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
        check_compatibility,
    )
    from brainscore_core.extraction_cache import weight_fingerprint_scope
    from brainscore_core.memory import check_memory
    from brainscore_core.score_metadata import (
        infer_score_protocol,
        record_measures,
        requested_output_channels_for_score,
        stamp_score_metadata,
    )

    import time as _time
    # accept either an identifier string or an already-built object
    from brainscore_core.preflight import check_cache_directory
    check_cache_directory()
    if isinstance(benchmark_identifier, str):
        from brainscore.data.local import check_benchmark_assets
        check_benchmark_assets(benchmark_identifier)
    benchmark = (load_benchmark(benchmark_identifier)
                 if isinstance(benchmark_identifier, str) else benchmark_identifier)
    model = (load_model(model_identifier)
             if isinstance(model_identifier, str) else model_identifier)
    model_id = (model_identifier if isinstance(model_identifier, str)
                else _resolve_identifier(model, 'custom-model'))
    benchmark_id = (benchmark_identifier if isinstance(benchmark_identifier, str)
                    else _resolve_identifier(benchmark, 'custom-benchmark'))
    requested_channels = requested_output_channels_for_score(benchmark)

    compatibility = check_compatibility(model, benchmark)
    if check_mem:
        check_memory(model, benchmark)

    # A legacy domain benchmark (marked by its domain loader) receives the
    # permanent facade, which runs its calls as sessions through participate.
    from brainscore_core.legacy_facade import declared_visual_degrees, subject_for_benchmark
    candidate = subject_for_benchmark(model, benchmark,
                                      visual_degrees=declared_visual_degrees(model))

    _t0 = _time.time()
    with weight_fingerprint_scope(), record_measures() as measures:
        result = benchmark(candidate)
    result.attrs['runtime_sec'] = round(_time.time() - _t0, 2)
    result.attrs['model_identifier'] = model_id
    # Benchmarks can supply a qualified identity for settings or data subsets.
    recorded_id = result.attrs.get('benchmark_identifier', benchmark_id)
    resolved_id = _resolve_identifier(benchmark, benchmark_id)
    if not isinstance(recorded_id, str) or recorded_id not in (benchmark_id, resolved_id):
        raise ValueError(
            'Score benchmark_identifier must match the requested alias or resolved benchmark identifier'
        )
    result.attrs.setdefault('benchmark_identifier', benchmark_id)
    stamp_score_metadata(
        result,
        model,
        requested_channels=requested_channels,
        protocol=infer_score_protocol(model, benchmark, requested_channels,
                                      measures=measures),
        harness_id='brainscore',
        measures=measures,
    )
    result.attrs['compatibility'] = compatibility.as_attrs()
    return result


# Populate unified registries once at import time
_populate_unified_registries()
