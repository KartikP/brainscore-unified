"""The MajajHong2015 and Pereira2018 plugins of unified: registration and route.

The two plugins reuse the legacy vision and language benchmarks' data, metric
and ceiling. Here those loaders are fakes, so nothing is downloaded: the tests
check which class each identifier builds, how the loader prepares it, and that
the ``-unified`` variants reach the subject through ``process()``.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import brainscore
import brainscore.benchmarks  # noqa: F401  (registers the unified benchmarks)
from brainscore_core.legacy_facade import LEGACY_API, subject_for_benchmark
from brainscore_core.metrics import Score

pytestmark = pytest.mark.unit

MAJAJHONG = {
    'MajajHong2015.V4-pls-unified': ('V4', 'MajajHong2015.V4-pls-unified'),
    'MajajHong2015.IT-pls-unified': ('IT', 'MajajHong2015.IT-pls-unified'),
    'MajajHong2015public.V4-pls-unified': ('V4', 'MajajHong2015.V4.public-pls-unified'),
    'MajajHong2015public.IT-pls-unified': ('IT', 'MajajHong2015.IT.public-pls-unified'),
}
PEREIRA = {  # registry identifier -> (built identifier, metric, unified route, cross-validation)
    f'Pereira2018.{experiment}-{variant}': (
        f'Pereira2018.{experiment}-{variant}', metric, unified, crossvalidation)
    for experiment in ('243sentences', '384sentences')
    for variant, metric, unified, crossvalidation in (
        ('linear', 'linear_pearsonr', False, None),
        ('linear-unified', 'linear_pearsonr', True, None),
        ('ridge-unified', 'ridge_pearsonr', True,
         {'split_coord': 'story', 'kfold': 'group', 'random_state': 1234}),
    )
}


def test_the_identifiers_are_registered_in_unified():
    assert set(MAJAJHONG) | set(PEREIRA) <= set(brainscore.benchmark_registry)


# --- MajajHong2015 -----------------------------------------------------------

class _Assembly:
    """The parts of a NeuroidAssembly the benchmark reads at construction."""
    dims = ('presentation', 'neuroid')

    def __init__(self, region):
        self.coords = {'region': np.array([region, region]),
                       'time_bin': SimpleNamespace(values=(70, 170))}
        self.stimulus_set = 'stimuli'

    def __getitem__(self, name):
        return self.coords[name]


@pytest.fixture
def majajhong(monkeypatch):
    from brainscore.benchmarks.majajhong2015 import benchmark
    loads = []

    def load_assembly(average_repetitions, region, access):
        loads.append((average_repetitions, region, access))
        return _Assembly(region)
    monkeypatch.setattr(benchmark, 'load_assembly', load_assembly)
    monkeypatch.setattr(benchmark, 'load_metric', lambda name, **kwargs: (name, kwargs))
    monkeypatch.setattr(benchmark, 'load_ceiling', lambda name: lambda assembly: Score(0.81))
    return benchmark, loads


@pytest.mark.parametrize('identifier', sorted(MAJAJHONG))
def test_majajhong_builds_the_moved_class(majajhong, identifier):
    module, loads = majajhong
    region, built = MAJAJHONG[identifier]
    access = 'public' if 'public' in identifier else 'private'
    benchmark = brainscore.load_benchmark(identifier)
    assert type(benchmark) is module.UnifiedNeuralBenchmark
    assert type(benchmark).__module__ == 'brainscore.benchmarks.majajhong2015.benchmark'
    assert benchmark.identifier == built and benchmark.version == 4
    assert benchmark.region == region and benchmark.parent == region
    assert benchmark._similarity_metric == ('pls', {'crossvalidation_kwargs': {
        'stratification_coord': 'object_name'}})
    assert loads == [(False, region, access), (True, region, access)]
    # Prepared as the vision loader prepares a benchmark.
    assert benchmark.required_modalities == {'vision'}
    assert benchmark.subject_api == 'typed'


def test_majajhong_reaches_the_subject_through_process(majajhong, monkeypatch):
    module, _ = majajhong
    import brainscore_vision.benchmark_helpers.neural_common as neural_common
    calls = []

    class Subject:
        def visual_degrees(self):
            return 8

        def start_recording(self, region, time_bins):
            calls.append(('start_recording', region, time_bins))

        def process(self, stimuli):
            calls.append(('process', stimuli))
            return SimpleNamespace(dims=('presentation', 'neuroid'))

    monkeypatch.setattr(module, 'place_on_screen', lambda stimuli, **kwargs: (stimuli, kwargs))
    monkeypatch.setattr(neural_common, 'explained_variance', lambda raw, ceiling: (raw, ceiling))
    benchmark = brainscore.load_benchmark('MajajHong2015public.IT-pls-unified')
    benchmark._similarity_metric = lambda source, target: Score(0.5)
    subject = Subject()
    assert subject_for_benchmark(subject, benchmark) is subject   # no legacy facade
    raw, ceiling = benchmark(subject)
    assert float(raw) == 0.5 and float(ceiling) == 0.81
    assert calls == [('start_recording', 'IT', [(70, 170)]),
                     ('process', ('stimuli', {'target_visual_degrees': 8,
                                              'source_visual_degrees': 8}))]


# --- Pereira2018 -------------------------------------------------------------

@pytest.fixture
def pereira(monkeypatch):
    from brainscore_language.benchmarks.pereira2018.benchmark import _Pereira2018Experiment
    from brainscore.benchmarks.pereira2018 import benchmark
    constructed = []

    def construct(self, experiment, metric, ceiling_s3_kwargs, crossvalidation_kwargs=None):
        constructed.append(dict(experiment=experiment, metric=metric,
                                crossvalidation_kwargs=crossvalidation_kwargs,
                                ceiling_s3_kwargs=ceiling_s3_kwargs))
        self._identifier = f"Pereira2018.{experiment}-{metric.split('_')[0]}"
        self._version = 1
    monkeypatch.setattr(_Pereira2018Experiment, '__init__', construct)
    return _Pereira2018Experiment, benchmark, constructed


@pytest.mark.parametrize('identifier', sorted(PEREIRA))
def test_pereira_builds_the_moved_class(pereira, identifier):
    legacy_class, module, constructed = pereira
    built, metric, unified, crossvalidation = PEREIRA[identifier]
    benchmark = brainscore.load_benchmark(identifier)
    expected = module._Pereira2018ExperimentUnified if unified else legacy_class
    assert type(benchmark) is expected
    assert benchmark.identifier == built
    assert benchmark.version == (2 if unified else 1)
    (arguments,) = constructed
    assert arguments['experiment'] == identifier.split('.')[1].split('-')[0]
    assert arguments['metric'] == metric
    assert arguments['crossvalidation_kwargs'] == crossvalidation
    assert arguments['ceiling_s3_kwargs']['sha1']   # the stored ceiling of that variant
    # Prepared as the language loader prepares a benchmark: the plain linear
    # variant receives the legacy facade, the unified ones the subject itself.
    assert benchmark.required_modalities == {'text'}
    assert benchmark.subject_api == ('typed' if unified else LEGACY_API)


def test_the_linear_variants_share_their_ceiling(pereira):
    _, _, constructed = pereira
    for experiment in ('243sentences', '384sentences'):
        brainscore.load_benchmark(f'Pereira2018.{experiment}-linear')
        brainscore.load_benchmark(f'Pereira2018.{experiment}-linear-unified')
    assert constructed[0]['ceiling_s3_kwargs'] == constructed[1]['ceiling_s3_kwargs']
    assert constructed[2]['ceiling_s3_kwargs'] == constructed[3]['ceiling_s3_kwargs']
