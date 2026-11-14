"""Pereira2018 variants that call the subject's ``process()``, and the plain
linear variants they are compared against.

The data, metric and ceiling are the legacy language benchmark's: this plugin
imports them from ``brainscore_language.benchmarks.pereira2018``. The ridge
variants group cross-validation by story; the linear ones split sentences at
random, which leaks within a passage, and upstream retired the plain linear
registration for that reason. Each is prepared as the language loader prepares
its benchmarks: the text modality, no channels, and the calling convention
(the plain linear variants receive the legacy facade).
"""
from brainscore import benchmark_registry


def _load(factory: str):
    def build():
        from brainscore_core.compatibility import ensure_legacy_benchmark_modalities
        from brainscore_core.legacy_facade import mark_legacy_benchmark
        from . import benchmark
        built = getattr(benchmark, factory)()
        return mark_legacy_benchmark(ensure_legacy_benchmark_modalities(built, {'text'}))
    return build


for _experiment in ('243sentences', '384sentences'):
    benchmark_registry[f'Pereira2018.{_experiment}-linear'] = _load(f'Pereira2018_{_experiment}')
    benchmark_registry[f'Pereira2018.{_experiment}-linear-unified'] = _load(
        f'Pereira2018_{_experiment}_unified')
    benchmark_registry[f'Pereira2018.{_experiment}-ridge-unified'] = _load(
        f'Pereira2018_{_experiment}_ridge_unified')
