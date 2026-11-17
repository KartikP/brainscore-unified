"""Register the Sun 2025 stimuli, human responses and local builder; no I/O at import."""
from brainscore import data_builder_registry, data_registry, stimulus_set_registry
from brainscore.data import local
from brainscore.data.preparation import DataBuilder

IDENTIFIER = 'Sun2025-second-order-motion'

PREPARED = local.register(local.LocalAsset(
    name='sun2025-prepared',
    env_var='BRAINSCORE_SUN2025_ROOT',
    default_path='.brainio/sun2025',
    kind='stimuli',
    why_local='The clips are reconstructed from frames in a 9.9 GB public archive; '
              'Brain-Score hosts no copy.',
    source='https://github.com/anoymized/multi-order-motion-model (revision c5886559) and '
           'https://huggingface.co/datasets/sunana/mateiral-controlled-motion-dataset (revision a7e83195)',
    obtain='Fetch the source frames, trial table and the three participant files with '
           'python -m brainscore.data.sun2025.data_packaging.fetch <new source directory>',
    prepare='python -m brainscore.data prepare Sun2025-second-order-motion '
            '--source <source directory> --output <new build directory>',
))


def _builder() -> DataBuilder:
    from .data_packaging.prepare import build
    return DataBuilder(build=build)


def _dataset(root=None):
    from .data import load_dataset
    return load_dataset(root)


def _stimuli(root=None):
    from .data import load_stimulus_set
    return load_stimulus_set(root)


data_builder_registry[IDENTIFIER] = _builder
data_registry[IDENTIFIER] = _dataset
stimulus_set_registry[IDENTIFIER] = _stimuli
