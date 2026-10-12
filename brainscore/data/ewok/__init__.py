"""Register EWoK data and its local builder without reading protected text."""
from brainscore import data_registry, stimulus_set_registry
from brainscore.data.preparation import DataBuilder, data_builder_registry

IDENTIFIER = 'EWoK-core-1.0'


def _builder() -> DataBuilder:
    from .prepare import build, resolve
    return DataBuilder(build=build, resolve=resolve)


def _dataset(root=None):
    from .data import load_dataset
    return load_dataset(root)


def _stimuli(root=None):
    from .data import load_stimulus_set
    return load_stimulus_set(root)


data_builder_registry[IDENTIFIER] = _builder
data_registry[IDENTIFIER] = _dataset
stimulus_set_registry[IDENTIFIER] = _stimuli
