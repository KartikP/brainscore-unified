"""The hosted LeBel2023-UTS03 files: one responses assembly shared by two stimulus sets.

Built with the registered default settings by ``data_packaging/package_hosted.py``.
Each file is fetched once, checked against its sha1 and cached locally.
"""
from typing import Optional

BUCKET = 'brainscore-storage'
FOLDER = 'brainscore-language/data/LeBel2023-UTS03'   # datasets live under <domain>/data/, not benchmarks/
ASSEMBLY_ID = 'LeBel2023-UTS03'
TR_STIMULI_ID = 'LeBel2023-UTS03'
WORD_STIMULI_ID = 'LeBel2023-UTS03-words'

# From package_hosted.py's output for the upload of 2026-11-04.
SHA1: dict[str, Optional[str]] = {
    'assembly': 'e03fd25503004d82ea932e272dbbdcb867d99a92',
    'TR stimuli': '9ddf67c7f3f9702473248eb15a877377d11732e6',
    'word stimuli': 'ce3d1ba0dc7457d189547ca8cda2c87ebcc8d5c8',
}
VERSION_ID: dict[str, Optional[str]] = {
    'assembly': 'sqH77RYfzyxn4KGU7TKNgmxg23E4iH0B',
    'TR stimuli': 'dcla2zdqjR08vrjBUeqWoEUCrBniO_2k',
    'word stimuli': 'MfKR5NqpBCCHkYW.CBfAQMVcvGDFeUCB',
}


def available() -> bool:
    return all(SHA1.values()) and all(VERSION_ID.values())


def _fetch(kind: str, identifier: str, extension: str, prefix: str):
    from brainscore_core.supported_data_standards.brainio.s3 import get_path
    return get_path(identifier, extension, BUCKET, VERSION_ID[kind], SHA1[kind],
                    filename_prefix=prefix, folder_name=FOLDER)


def _stimulus_set(kind: str, identifier: str, name: str):
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    # A context such as "NA" or "null" is text, not a missing value, and word onset
    # times must read back exactly as written.
    stimuli = StimulusSet(pd.read_csv(_fetch(kind, identifier, 'csv', 'stimulus_'),
                                      keep_default_na=False, float_precision='round_trip'))
    stimuli.identifier = name
    stimuli.stimulus_paths = {}          # text only; no files to resolve
    return stimuli


def tr_stimulus_set():
    from .data import DEFAULT_CONTEXT_TR, IDENTIFIER
    return _stimulus_set('TR stimuli', TR_STIMULI_ID, f'{IDENTIFIER}-context{DEFAULT_CONTEXT_TR}tr')


def word_stimulus_set():
    from .data import DEFAULT_CONTEXT_WORDS, IDENTIFIER
    return _stimulus_set('word stimuli', WORD_STIMULI_ID, f'{IDENTIFIER}-words{DEFAULT_CONTEXT_WORDS}')


def assembly():
    import xarray as xr
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    from .data import IDENTIFIER, TR_SEC
    responses = NeuroidAssembly(xr.open_dataarray(_fetch('assembly', ASSEMBLY_ID, 'nc', 'assy_')).load())
    responses.name = IDENTIFIER
    responses.attrs['tr_sec'] = TR_SEC
    return responses
