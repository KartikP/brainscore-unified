"""Register the Sun 2025 stimuli, human responses and builder; no I/O at import."""
from brainscore import data_builder_registry, data_registry, stimulus_set_registry
from brainscore.data import local
from brainscore.data.packaging import DataBuilder

IDENTIFIER = 'Sun2025-second-order-motion'
_SOURCE = ('https://github.com/anoymized/multi-order-motion-model (revision c5886559, Apache-2.0) and '
           'https://huggingface.co/datasets/sunana/mateiral-controlled-motion-dataset (revision a7e83195, MIT)')
_PACKAGE = 'python -m brainscore.data package sun2025'

PREPARED = local.register(local.LocalAsset(
    name='sun2025-prepared',
    kind='stimuli',
    why_local='The clips are reconstructed from frames in a 9.9 GB public archive; '
              'Brain-Score hosts no copy.',
    source=_SOURCE,
    obtain='Nothing to download by hand: the command below fetches the source frames, the trial table '
           'and the three participant files, and writes the clips and human responses.',
    prepare=_PACKAGE,
    used_by=['Sun2025-motion-vector-error'],
))

data_builder_registry['sun2025'] = lambda: DataBuilder(
    source=_SOURCE,
    access='public, no account.',
    layout='none: the script fetches the pinned files itself. A folder that holds an earlier fetch '
           '(`source-manifest.json` and the files it lists, written by '
           '`python -m brainscore.data.sun2025.data_packaging.fetch <new folder>`) is read in its place.',
    needs='numpy, pandas, openpyxl and opencv-python with the FFV1 encoder; the PNG frames of 40 scenes, '
          'read in ranges from the 9.9 GB archive (the archive is not stored); a compute machine, not a laptop.',
    writes='`clips/` (1,120 lossless FFV1 clips), `stimuli.json`, `human-responses.json`, '
           '`source-manifest.json` and `manifest.json` with the SHA-256 of every file.',
    reads=PREPARED.name,
    build='python -m brainscore.data.sun2025.data_packaging.prepare --source {source} --work {work} --out {output}',
    fetches=True,
)


def _dataset(root=None):
    from .data import load_dataset
    return load_dataset(root)


def _stimuli(root=None):
    from .data import load_stimulus_set
    return load_stimulus_set(root)


data_registry[IDENTIFIER] = _dataset
stimulus_set_registry[IDENTIFIER] = _stimuli
