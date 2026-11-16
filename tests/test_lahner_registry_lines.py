"""Each registered Lahner variant is one registry line that names its full identifier; the
time-resolved class has no registry line."""
import functools

import pytest

import brainscore
import brainscore.benchmarks  # noqa: F401  (registers the unified benchmarks)
from brainscore.benchmarks.lahner2024 import Lahner2024BOLDMoments, Lahner2024BOLDMoments_timeresolved

LAHNER = sorted(key for key in brainscore.benchmark_registry if key.startswith('Lahner2024-'))


def test_the_glm_variants_are_registered_and_no_time_resolved_one():
    # The time-resolved class is not registered (see `lahner2024/__init__.py`).
    assert not [key for key in brainscore.benchmark_registry if 'timeresolved' in key]
    glm = {'Lahner2024-fMRI-naturalistic' + suffix for suffix in (
        '', '-visualROI', '-multimodal', '-multimodal-visualROI', '-multimodal-auditoryROI')}
    assert glm <= set(LAHNER), f'GLM variants no longer registered: {sorted(glm - set(LAHNER))}'


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
