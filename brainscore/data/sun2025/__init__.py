"""Register Sun 2025 stimuli, human responses and source packaging lazily."""
from brainscore import data_registry, stimulus_set_registry
from brainscore.data.preparation import DataBuilder, data_builder_registry

IDENTIFIER = 'Sun2025-second-order-motion'


def _dataset(root=None):
    from .data import load_dataset
    return load_dataset(root)


def _stimuli(root=None):
    from .data import load_stimulus_set
    return load_stimulus_set(root)


def _builder():
    from .data_packaging.prepare import build
    return DataBuilder(build=build)


data_registry[IDENTIFIER] = _dataset
stimulus_set_registry[IDENTIFIER] = _stimuli
data_builder_registry[IDENTIFIER] = _builder
