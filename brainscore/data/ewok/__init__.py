"""Register EWoK data and its local builder without reading protected text."""
from brainscore import data_builder_registry, data_registry, stimulus_set_registry
from brainscore.data import local
from brainscore.data.preparation import DataBuilder

IDENTIFIER = 'EWoK-core-1.0'

EWOK_ROOT = local.register(local.LocalAsset(
    name='ewok-core-1.0',
    env_var='BRAINSCORE_EWOK_ROOT',
    default_path='.brainio/ewok-core-1.0',
    kind='stimuli',
    why_local='The authors require protected or gated distribution of EWoK text.',
    source='https://huggingface.co/datasets/ewok-core/ewok-core-1.0',
    obtain='Obtain the native files from the authors, or use approved Hugging Face access.',
    prepare='python -m brainscore.data prepare EWoK-core-1.0 '
            '--source <native files> --output <new build directory>',
    used_by=['EWoK-core-1.0-logprobs', 'EWoK-core-1.0-choice'],
))


def _builder() -> DataBuilder:
    from .data_packaging.prepare import build, resolve
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
