"""LeBel 2023 story-listening fMRI (participant UTS03), whole cortex.

Source data: OpenNeuro ds003020 (LeBel et al., 2023, Scientific Data), CC0.

Registered, with no I/O until called:

- dataset ``LeBel2023-UTS03``: responses, one row per retained TR, one column per vertex.
- stimulus set ``LeBel2023-UTS03``: one row per TR, the words heard in the preceding TRs.
- stimulus set ``LeBel2023-UTS03-words``: one row per spoken word, with its running context.

The default settings load hosted files (``hosted.py``). Other context settings, or a
build before the files are hosted, convert the source pickle named by
``BRAINSCORE_LEBEL_PICKLE`` (``data.py``).
"""

from brainscore import data_registry, stimulus_set_registry
from brainscore.data import local

# Declared before the imports below: ``data.py`` reads LEBEL_PICKLE.
LEBEL_PICKLE = local.register(local.LocalAsset(
    name='lebel2023-pickle',
    env_var='BRAINSCORE_LEBEL_PICKLE',
    default_path='Downloads/assembly_lebel_uts03.pkl',
    kind='neural data',
    why_local='Needed only for non-default context settings; the default files are '
              'hosted (brainscore/data/lebel2023/hosted.py).',
    source='https://github.com/GT-LIT-Lab/litcoder_release',
    obtain='The UTS03 assembly pickle for LeBel et al. 2023.',
    used_by=[],      # every registered LeBel variant reads the hosted files
))

LANA_ATLAS = local.register(local.LocalAsset(
    name='lana-atlas',
    env_var='BRAINSCORE_LANA_ATLAS',
    default_path='Downloads/20425209/FS',
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
