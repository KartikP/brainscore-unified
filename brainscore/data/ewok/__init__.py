"""Register EWoK data and its local builder without reading protected text."""
from brainscore import data_builder_registry, data_registry, stimulus_set_registry
from brainscore.data import local
from brainscore.data.packaging import DataBuilder

IDENTIFIER = 'EWoK-core-1.0'

EWOK_ROOT = local.register(local.LocalAsset(
    name='ewok-core-1.0',
    env_var='BRAINSCORE_EWOK_ROOT',
    default_path='.brainio/ewok-core-1.0',
    kind='stimuli',
    why_local='The authors require protected or gated distribution of EWoK text.',
    source='https://huggingface.co/datasets/ewok-core/ewok-core-1.0',
    obtain='Obtain the native files from the authors, or use approved Hugging Face access.',
    prepare='python -m brainscore.data package ewok',
    used_by=['EWoK-core-1.0-logprobs', 'EWoK-core-1.0-choice'],
))

data_builder_registry['ewok'] = DataBuilder(
    source='EWoK-core-1.0 (Ivanova et al. 2024, https://arxiv.org/abs/2405.09605): '
           'https://huggingface.co/datasets/ewok-core/ewok-core-1.0 at revision '
           '34d912a608066c92e2990a0328ffc3bd9a716042, or the paper checkout '
           'https://github.com/ewok-core/ewok-paper at revision 9e40d30e242925866ee50448f80a13bcdf971318. '
           'Licence: the authors\' terms of use '
           '(https://github.com/ewok-core/ewok-paper/blob/main/TERMS_OF_USE.txt); do not publish the text.',
    access='gated: request access on the Hugging Face page, then sign in with `hf auth login`. '
           'The paper checkout needs the password of the terms of use in EWOK_ARCHIVE_PASSWORD.',
    layout='the native table as `hf download ewok-core/ewok-core-1.0 data/test/ewok-core-1.0.parquet '
           '--repo-type dataset --revision 34d912a608066c92e2990a0328ffc3bd9a716042 --local-dir <source folder>` '
           'writes it (native CSV or Parquet files also work), or the paper checkout with '
           '`analyses/data.zip` and `config.zip`.',
    needs='`pip install -e \'.[ewok-data]\'` (file readers and the `hf` command); under 32 MB; minutes.',
    writes='`items.json` (one item per contrast), `manifest.json` (counts and the items checksum), '
           '`build.json`. Use it: set BRAINSCORE_EWOK_ROOT to the output folder.',
    build='python -m brainscore.data.ewok.data_packaging.prepare {source} {output}',
)


def _dataset(root=None):
    from .data import load_dataset
    return load_dataset(root)


def _stimuli(root=None):
    from .data import load_stimulus_set
    return load_stimulus_set(root)


data_registry[IDENTIFIER] = _dataset
stimulus_set_registry[IDENTIFIER] = _stimuli
