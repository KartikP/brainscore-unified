"""The Pereira2018 plugin of unified: the retired linear variants.

The plugin reuses the legacy language benchmark's data, metric and ceiling. Here those
loaders are fakes, so nothing is downloaded: the tests check which class each identifier
builds and how the loader prepares it (the legacy facade, as for every legacy benchmark).
"""
import pytest

import brainscore
import brainscore.benchmarks  # noqa: F401  (registers the unified benchmarks)
from brainscore_core.legacy_facade import LEGACY_API

pytestmark = pytest.mark.unit

PEREIRA = {  # registry identifier -> (built identifier, metric)
    f'Pereira2018.{experiment}-linear': (f'Pereira2018.{experiment}-linear', 'linear_pearsonr')
    for experiment in ('243sentences', '384sentences')
}


def test_the_identifiers_are_registered_in_unified():
    assert set(PEREIRA) <= set(brainscore.benchmark_registry)


def test_no_unified_copy_of_a_legacy_benchmark_is_registered():
    """The legacy MajajHong2015 and Pereira2018 benchmarks reach every subject through the facade."""
    assert not [key for key in brainscore.benchmark_registry if key.endswith('-unified')]


@pytest.fixture
def pereira(monkeypatch):
    from brainscore_language.benchmarks.pereira2018.benchmark import _Pereira2018Experiment
    constructed = []

    def construct(self, experiment, metric, ceiling_s3_kwargs, crossvalidation_kwargs=None):
        constructed.append(dict(experiment=experiment, metric=metric,
                                crossvalidation_kwargs=crossvalidation_kwargs,
                                ceiling_s3_kwargs=ceiling_s3_kwargs))
        self._identifier = f"Pereira2018.{experiment}-{metric.split('_')[0]}"
        self._version = 1
    monkeypatch.setattr(_Pereira2018Experiment, '__init__', construct)
    return _Pereira2018Experiment, constructed


@pytest.mark.parametrize('identifier', sorted(PEREIRA))
def test_pereira_builds_the_legacy_class(pereira, identifier):
    legacy_class, constructed = pereira
    built, metric = PEREIRA[identifier]
    benchmark = brainscore.load_benchmark(identifier)
    assert type(benchmark) is legacy_class
    assert benchmark.identifier == built
    assert benchmark.version == 1
    (arguments,) = constructed
    assert arguments['experiment'] == identifier.split('.')[1].split('-')[0]
    assert arguments['metric'] == metric
    assert arguments['crossvalidation_kwargs'] is None
    assert arguments['ceiling_s3_kwargs']['sha1']   # the stored ceiling of that variant
    # Prepared as the language loader prepares a benchmark: the subject gets the facade.
    assert benchmark.required_modalities == {'text'}
    assert benchmark.subject_api == LEGACY_API
