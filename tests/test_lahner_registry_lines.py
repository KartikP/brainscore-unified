"""Each Lahner variant is one registry line that names its full identifier."""
import functools

import pytest

import brainscore
import brainscore.benchmarks  # noqa: F401  (registers the unified benchmarks)
from brainscore.benchmarks.lahner2024 import Lahner2024BOLDMoments, Lahner2024BOLDMoments_timeresolved

LAHNER = sorted(key for key in brainscore.benchmark_registry if key.startswith('Lahner2024-'))


def test_the_lahner_variants_are_the_twelve_registered_before():
    assert len(LAHNER) == 12 and 'Lahner2024-fMRI-naturalistic-timeresolved-multimodal-auditoryROI' in LAHNER


@pytest.mark.parametrize('key', LAHNER)
def test_a_registry_key_is_the_identifier_its_line_passes(key):
    # Not built: the constructors load data. The line is read, not run.
    factory = brainscore.benchmark_registry[key]
    assert isinstance(factory, functools.partial)
    assert factory.keywords['identifier'] == key
    assert 'identifier_suffix' not in factory.keywords


@pytest.mark.parametrize('benchmark_class', [Lahner2024BOLDMoments, Lahner2024BOLDMoments_timeresolved])
def test_a_lahner_class_needs_an_identifier(benchmark_class):
    with pytest.raises(TypeError, match='identifier'):
        benchmark_class()
