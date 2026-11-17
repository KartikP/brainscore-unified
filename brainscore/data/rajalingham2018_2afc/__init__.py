"""Data plugin for the Rajalingham2018 2-AFC scoring library (``brainscore.tools.rajalingham2018_2afc``).

``trials.csv`` holds the (image, distractor) conditions and ``tokens.csv`` the object token images.
The human responses come from the hosted ``Rajalingham2018.public`` dataset. This plugin registers nothing.
"""
from pathlib import Path

from brainscore import data_builder_registry
from brainscore.data.packaging import DataBuilder

FOLDER = Path(__file__).resolve().parent
TRIALS_CSV = FOLDER / 'trials.csv'
TOKENS_CSV = FOLDER / 'tokens.csv'

data_builder_registry['rajalingham2018-2afc'] = lambda: DataBuilder(
    source='the Brain-Score vision datasets Rajalingham2018.public (human 2-AFC trials of '
           'Rajalingham et al. 2018) and objectome.public (the images); licence as the vision plugins state it.',
    access='public, no account: the hosted vision files download with no account.',
    layout='none: the script loads the two vision datasets through brainscore_vision.',
    needs='brainscore_vision; a small download; under a minute.',
    writes='`trials.csv` (one row per test image and distractor) and `tokens.csv` (one token image per '
           'object). The scoring tool reads the two files that ship in the plugin folder '
           '`brainscore/data/rajalingham2018_2afc/`, not this output; the check compares the rebuild with '
           'them, and there is nothing to put elsewhere.',
    build='python -m brainscore.data.rajalingham2018_2afc.data_packaging.build_trials --out {output}',
    sha1={'trials.csv': '71d7317315c7eb310bf25489d81cdebfdd2cdafd',
          'tokens.csv': '18f219fd4aa96567377b196b8f165b5c6996bc59'},
    fetches=True,
)
