"""LeBel 2023 story-listening fMRI (participant UTS03), whole cortex.

Source data: OpenNeuro ds003020 (LeBel et al., 2023, Scientific Data), CC0.

Registered, with no I/O until called:

- dataset ``LeBel2023-UTS03``: responses, one row per retained TR, one column per vertex.
- stimulus set ``LeBel2023-UTS03``: one row per TR, the words heard in the preceding TRs.
- stimulus set ``LeBel2023-UTS03-words``: one row per spoken word, with its running context.

The default settings load hosted files (``hosted.py``). Other context settings, or a
build before the files are hosted, convert the source pickle in the source folder of
``package_data('lebel2023')`` (``data.py``).
"""

from brainscore import data_builder_registry, data_registry, stimulus_set_registry
from brainscore.data import local
from brainscore.data.packaging import DataBuilder

LANA_ATLAS = local.register(local.LocalAsset(
    name='lana-atlas',
    kind='atlas',
    why_local='Fedorenko-lab resource distributed under its own terms.',
    source='https://osf.io/kzwbh/  (doi:10.17605/OSF.IO/KZWBH)',
    obtain='The "FS Atlas" archive - the FreeSurfer-surface arm, which needs no '
           'volume-to-surface projection. Unzip it; the directory holding '
           'LH_LanA_n804.nii.gz and RH_LanA_n804.nii.gz is what to point at.',
    used_by=['LeBel2023-UTS03-encoding-languagemask'],
))

from . import hosted  # noqa: E402
from .data import DEFAULT_CONTEXT_TR, DEFAULT_CONTEXT_WORDS, IDENTIFIER, build_word_level, load  # noqa: E402

WORDS_IDENTIFIER = f'{IDENTIFIER}-words'

data_builder_registry['lebel2023'] = lambda: DataBuilder(
    source='LeBel et al. 2023, OpenNeuro ds003020 (https://openneuro.org/datasets/ds003020), CC0, as one '
           'file: `assembly_lebel_uts03.pkl` (2,975,580,003 bytes, sha1 d672e6fa0413259cb761042bd0a64990b12ab6dc), '
           'a `SimpleNeuroidAssembly` pickle of the Huth-lab `encoding` package for participant UTS03 (25 stories). '
           'How it was made is not recorded. The LITCoder release (https://github.com/GT-LIT-Lab/litcoder_release) '
           'does not offer this file: it offers a script (`fmri_processing/to_mni_lebel.py`) that makes '
           'per-participant pickles from fMRIPrep output.',
    access='the source file is not downloadable from a public page; a Brain-Score maintainer keeps an archive '
           'copy (the sha1 above): ask the maintainers. The recordings themselves are public (CC0).',
    layout='`assembly_lebel_uts03.pkl` (sha1 d672e6fa0413259cb761042bd0a64990b12ab6dc).',
    needs='the unified install (numpy, pandas, xarray); about 3 GB of disk for the pickle and more '
          'than 3 GB of memory; time not measured.',
    writes=f'`assy_{hosted.ASSEMBLY_ID}.nc`, `stimulus_{hosted.TR_STIMULI_ID}.csv`, '
           f'`stimulus_{hosted.WORD_STIMULI_ID}.csv`, with the registered default settings.',
    reads='brainio',
    build='python -m brainscore.data.lebel2023.data_packaging.package_hosted '
          '--pickle {source}/assembly_lebel_uts03.pkl --out {output}',
    requires=('assembly_lebel_uts03.pkl',),
    sha1={f'assy_{hosted.ASSEMBLY_ID}.nc': hosted.SHA1['assembly'],
          f'stimulus_{hosted.TR_STIMULI_ID}.csv': hosted.SHA1['TR stimuli'],
          f'stimulus_{hosted.WORD_STIMULI_ID}.csv': hosted.SHA1['word stimuli']},
)

data_builder_registry['lana-atlas'] = lambda: DataBuilder(
    source=f'the LanA language atlas (Lipkin et al. 2022), {LANA_ATLAS.source}; its own terms.',
    access='public, no account.',
    layout=LANA_ATLAS.obtain,
    needs='a download, size not measured; no tools.',
    writes='The atlas is read as published. Put the folder at `$BRAINSCORE_DATA_HOME/lana-atlas` '
           '(`~/brainscore-data/lana-atlas` by default), or put a link there to it.',
)


def _responses():
    if hosted.available():
        return hosted.assembly()
    return load()[1]


def _tr_stimuli(context_window_tr: int = DEFAULT_CONTEXT_TR):
    if hosted.available() and context_window_tr == DEFAULT_CONTEXT_TR:
        return hosted.tr_stimulus_set()
    return load(context_window_tr=context_window_tr)[0]


def _word_stimuli(context_words: int = DEFAULT_CONTEXT_WORDS):
    if hosted.available() and context_words == DEFAULT_CONTEXT_WORDS:
        return hosted.word_stimulus_set()
    return build_word_level(context_words=context_words)[0]


data_registry[IDENTIFIER] = _responses
stimulus_set_registry[IDENTIFIER] = _tr_stimuli
stimulus_set_registry[WORDS_IDENTIFIER] = _word_stimuli
