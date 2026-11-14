"""MajajHong2015 variants that call the subject's ``process()``.

The data, metric and ceiling are the legacy vision benchmark's: this plugin
imports them from ``brainscore_vision.benchmarks.majajhong2015`` and changes
only how the stimuli reach the subject. Each is prepared as the vision loader
prepares its benchmarks: the vision modality, no channels, and the typed
calling convention the class names.
"""
from brainscore import benchmark_registry


def _load(region: str, access: str):
    def build():
        from brainscore_core.compatibility import ensure_legacy_benchmark_modalities
        from brainscore_core.legacy_facade import mark_legacy_benchmark
        from .benchmark import _build_majajhong_unified
        benchmark = _build_majajhong_unified(region=region, access=access)
        return mark_legacy_benchmark(ensure_legacy_benchmark_modalities(benchmark, {'vision'}))
    return build


benchmark_registry['MajajHong2015.V4-pls-unified'] = _load('V4', 'private')
benchmark_registry['MajajHong2015.IT-pls-unified'] = _load('IT', 'private')
benchmark_registry['MajajHong2015public.V4-pls-unified'] = _load('V4', 'public')
benchmark_registry['MajajHong2015public.IT-pls-unified'] = _load('IT', 'public')
