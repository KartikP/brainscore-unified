from functools import partial

from brainscore import benchmark_registry


def _load(**configuration):
    from .benchmark import StLaurent2026Things
    return StLaurent2026Things(**configuration)


# One line per region, the participants pooled (sub-01, 02, 03: the participants with their own pRF and
# fLoc localizer masks). V4 is the human homolog hV4 (pRF mask). See benchmark.py.
benchmark_registry['StLaurent2026-fMRI-things-V1'] = partial(
    _load, region='V1', identifier='StLaurent2026-fMRI-things-V1')
benchmark_registry['StLaurent2026-fMRI-things-V2'] = partial(
    _load, region='V2', identifier='StLaurent2026-fMRI-things-V2')
benchmark_registry['StLaurent2026-fMRI-things-V3'] = partial(
    _load, region='V3', identifier='StLaurent2026-fMRI-things-V3')
benchmark_registry['StLaurent2026-fMRI-things-V4'] = partial(
    _load, region='V4', identifier='StLaurent2026-fMRI-things-V4')
benchmark_registry['StLaurent2026-fMRI-things-FFA'] = partial(
    _load, region='FFA', identifier='StLaurent2026-fMRI-things-FFA')
benchmark_registry['StLaurent2026-fMRI-things-OFA'] = partial(
    _load, region='OFA', identifier='StLaurent2026-fMRI-things-OFA')
benchmark_registry['StLaurent2026-fMRI-things-pSTS'] = partial(
    _load, region='pSTS', identifier='StLaurent2026-fMRI-things-pSTS')
benchmark_registry['StLaurent2026-fMRI-things-PPA'] = partial(
    _load, region='PPA', identifier='StLaurent2026-fMRI-things-PPA')
benchmark_registry['StLaurent2026-fMRI-things-OPA'] = partial(
    _load, region='OPA', identifier='StLaurent2026-fMRI-things-OPA')
benchmark_registry['StLaurent2026-fMRI-things-MPA'] = partial(
    _load, region='MPA', identifier='StLaurent2026-fMRI-things-MPA')
benchmark_registry['StLaurent2026-fMRI-things-EBA'] = partial(
    _load, region='EBA', identifier='StLaurent2026-fMRI-things-EBA')
