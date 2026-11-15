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

from .brainscore_model import BrainScoreModel
from .selection import (
    CompositeSelector, FunctionalSelection, IndexSelection, LayerSelector,
    RandomSelection, UnitSelection, UnitSelector, selection_coordinates,
    selection_unit_indices,
)
from brainscore_core.contract import Subject
from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Metric, Score

_logger = logging.getLogger(__name__)

class _Registry(dict):
    """A registry that refuses a second registration under a key it holds.

    A replaced factory would silently change what is scored under that name,
    so a second assignment from another source raises and names both sources.
    The same source again (a module imported a second time) replaces quietly. Remove the entry
    first (``del registry[key]``) to replace it on purpose, as a test does.
    """

    def __init__(self) -> None:
        super().__init__()
        self._sources: Dict[str, str] = {}

    @staticmethod
    def _source_of(value, frame) -> str:
        """Name the source of a registration: the factory's module and qualified
        name, or the registering statement (file and line) for a lambda."""
        name = getattr(value, '__qualname__', None)
        if name and '<lambda>' not in name and getattr(value, '__module__', None):
            return f'{value.__module__}.{name}'
        return f'{frame.f_code.co_filename}:{frame.f_lineno}'

    def _register(self, key, value, frame) -> None:
        source = self._source_of(value, frame)
        if key in self and self._sources.get(key) != source:
            raise ValueError(
                f'{key!r} is already registered (by {self._sources.get(key, "unknown")}); '
                f'a second registration (by {source}) would silently replace it. '
                f'Use a new identifier.')
        super().__setitem__(key, value)       # the same source again (a re-import) replaces quietly
        self._sources[key] = source

    def __setitem__(self, key, value) -> None:
        import sys
        self._register(key, value, sys._getframe(1))

    def update(self, *args, **kwargs) -> None:
        import sys
        frame = sys._getframe(1)
        for key, value in dict(*args, **kwargs).items():
            self._register(key, value, frame)

    def setdefault(self, key, default=None):
        import sys
        if key not in self:
            self._register(key, default, sys._getframe(1))
        return self[key]

    def __ior__(self, other):
        import sys
        frame = sys._getframe(1)
        for key, value in dict(other).items():
            self._register(key, value, frame)
        return self

    def __delitem__(self, key) -> None:
        super().__delitem__(key)
        self._sources.pop(key, None)


model_registry: Dict[str, Callable[[], Subject]] = _Registry()
benchmark_registry: Dict[str, Callable[[], Benchmark]] = _Registry()
metric_registry: Dict[str, Callable[[], Metric]] = _Registry()
data_registry: Dict[str, Callable[..., Any]] = _Registry()
stimulus_set_registry: Dict[str, Callable[..., Any]] = _Registry()
# Lazy factories of local data builders (``brainscore.data.preparation.DataBuilder``).
data_builder_registry: Dict[str, Callable[[], Any]] = _Registry()


# Plugin packages that failed to import: {qualified name: 'ErrorType: message'}.
_plugin_import_failures: Dict[str, str] = {}


def _import_plugins(package: str) -> None:
    """Import every plugin of the kind folder ``package`` so that it registers.

    A plugin is a sub-package (a folder with an ``__init__.py``) whose name does not
    start with ``_``; a plain module (a shared helper) is not imported. A plugin whose
    import fails is named in a warning with the cause and recorded; the other plugins
    still register, and the loaders' not-found message names it.
    """
    import importlib
    import pkgutil
    import sys
    for info in sorted(pkgutil.iter_modules(sys.modules[package].__path__), key=lambda i: i.name):
        if not info.ispkg or info.name.startswith('_'):
            continue
        name = f'{package}.{info.name}'
        try:
            importlib.import_module(name)
        except ImportError as error:
            _plugin_import_failures[name] = f'{type(error).__name__}: {error}'
            _logger.warning(f'plugin {name} failed to import and registers nothing: '
                            f'{type(error).__name__}: {error}')


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


def _not_found(kind: str, identifier: str, sources: str, folder: str, listing: str = '') -> KeyError:
    """The one message of a loader that finds no identifier."""
    listed = f' {listing} lists them all.' if listing else ''
    failed = {name: cause for name, cause in _plugin_import_failures.items()
              if name.startswith(f'brainscore.{folder}.')}
    broken = (' These plugin packages failed to import, so their identifiers are missing: '
              + '; '.join(f'{name} ({cause})' for name, cause in sorted(failed.items())) + '.'
              if failed else '')
    return KeyError(
        f"{kind} '{identifier}' not found in {sources} registries.{listed}{broken} A plugin "
        f"registers when its package is imported: a folder brainscore/{folder}/<name>/ with an "
        f"__init__.py is imported with brainscore; a plugin in another package registers when "
        f"you import that package before you load it.")


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
    raise _not_found('Model', identifier, 'unified, vision, or language', 'models',
                     'brainscore.list_models()')


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
    raise _not_found('Benchmark', identifier, 'unified, vision, or language', 'benchmarks',
                     'brainscore.list_benchmarks()')


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

    raise _not_found('Metric', identifier, 'unified or vision', 'metrics')


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
    raise _not_found('Dataset', identifier, 'unified, vision, or language', 'data')


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
    raise _not_found('StimulusSet', identifier, 'unified or vision', 'data')


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
    """Score a subject on a benchmark.

    Each argument is either an identifier ``str`` — loaded from the unified
    registry (with domain fallbacks) — or an already-constructed object (a
    ``Subject`` / benchmark). Passing objects lets you score a subject you just
    built without registering it first::

        subject = BrainScoreModel('my-vlm', ...)
        score(subject, 'MajajHong2015public.IT-pls')

    then runs the benchmark on the subject.

    An incompatible pair, known from the declarations, returns the N/A result
    before any compute (``brainscore_core.score_status.na_score``: NaN,
    ``attrs['status'] == 'N/A'``, ``reason``, ``missing_channels``). A refusal
    when the run starts (a subject that refuses a session, a benchmark's own
    check) raises ``ExecutionNotFeasibleError``; a malformed declaration
    (``ValueError``) and a memory-check failure raise too.
    """
    from brainscore_core.compatibility import CompatibilityError, check_compatibility
    from brainscore_core.extraction_cache import weight_fingerprint_scope
    from brainscore_core.memory import check_memory
    from brainscore_core.score_status import na_score
    from brainscore_core.score_metadata import (
        requested_output_channels_for_score,
        stamp_score_metadata,
    )
    try:
        from brainscore_core.score_metadata import record_measures
    except ImportError as error:
        raise ImportError(
            'brainscore needs a brainscore_core with score_metadata.record_measures; '
            'install the core revision named in install/peer-revisions.json'
        ) from error

    import time as _time
    # accept either an identifier string or an already-built object
    from brainscore_core.preflight import check_cache_directory
    check_cache_directory()
    if isinstance(benchmark_identifier, str):
        from brainscore.data.local import check_benchmark_assets
        check_benchmark_assets(benchmark_identifier)
    benchmark = (load_benchmark(benchmark_identifier)
                 if isinstance(benchmark_identifier, str) else benchmark_identifier)
    subject = (load_model(model_identifier)
               if isinstance(model_identifier, str) else model_identifier)
    model_id = (model_identifier if isinstance(model_identifier, str)
                else _resolve_identifier(subject, 'custom-model'))
    benchmark_id = (benchmark_identifier if isinstance(benchmark_identifier, str)
                    else _resolve_identifier(benchmark, 'custom-benchmark'))
    requested_channels = requested_output_channels_for_score(benchmark)

    # The one place N/A is made: the declaration check, before any compute
    # (rules 5.6, 5.7). A refusal during the run below is an error.
    try:
        compatibility = check_compatibility(subject, benchmark)
    except CompatibilityError as error:
        result = na_score(error)
        result.attrs.update(model_identifier=model_id, benchmark_identifier=benchmark_id)
        return result
    if check_mem:
        check_memory(subject, benchmark)

    # A legacy domain benchmark (marked by its domain loader) receives the
    # permanent facade, which runs its calls as sessions through participate.
    from brainscore_core.legacy_facade import declared_visual_degrees, subject_for_benchmark
    received = subject_for_benchmark(subject, benchmark,
                                     visual_degrees=declared_visual_degrees(subject))

    _t0 = _time.time()
    # A session no note names belongs to this score's subject, also when this score runs inside another.
    with weight_fingerprint_scope(), record_measures(subject=getattr(subject, 'identifier', None)) as measures:
        result = benchmark(received)
    # The run must leave the subject as it found it: every intervention a session
    # of the benchmark applied is undone. Interventions the caller applied stay.
    measures.check_released(benchmark_id, model_id)
    result.attrs['runtime_sec'] = round(_time.time() - _t0, 2)
    result.attrs['model_identifier'] = model_id
    # A benchmark can supply a qualified identity for settings or data subsets.
    stamp_score_metadata(
        result,
        subject,
        requested_channels=requested_channels,
        harness_id='brainscore',
        measures=measures,
        benchmark=benchmark,
        benchmark_identifier=benchmark_id,
    )
    result.attrs['compatibility'] = compatibility.as_attrs()
    return result


# Populate unified registries once at import time
_populate_unified_registries()
