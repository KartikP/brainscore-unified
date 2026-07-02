"""Data plugin for ROAR / Yeatman2021 lexical-decision assets."""

from brainscore import data_registry, stimulus_set_registry


STIMULUS_SET_ID = 'Yeatman2021'
ASSEMBLY_ID = 'Yeatman2021'
BUCKET = 'brainscore-storage/brainscore-vision/benchmarks/Yeatman2021'
STIMULUS_CSV_SHA1 = '04fbe35ba4384cf8ba5d152961d21f278a77e32a'
STIMULUS_ZIP_SHA1 = '0adacf476500f8f8a4b0a6d34f8688b22b583fcd'
STIMULUS_CSV_VERSION_ID = 'R7ZDCbKJAb1..ltN609fBKGN2v6pQzwF'
STIMULUS_ZIP_VERSION_ID = '0M.L.ZJwaDZ1psldFkspKKcTvX5qS3r4'
ASSEMBLY_SHA1 = 'fee134858a68ac4bd113f6b6ec79f4f8722db1b8'
ASSEMBLY_VERSION_ID = 'zAOrOdPgYFoyklSkOXgqg.uW8oDqR7lj'


def load_yeatman2021_stimulus_set():
    """Load ROAR stimulus metadata and local image paths from S3."""
    from brainscore_core.supported_data_standards.brainio.s3 import (
        load_stimulus_set_from_s3,
    )

    stimulus_set = load_stimulus_set_from_s3(
        identifier=STIMULUS_SET_ID,
        bucket=BUCKET,
        csv_sha1=STIMULUS_CSV_SHA1,
        zip_sha1=STIMULUS_ZIP_SHA1,
        csv_version_id=STIMULUS_CSV_VERSION_ID,
        zip_version_id=STIMULUS_ZIP_VERSION_ID,
    )
    return _postprocess_stimulus_set(stimulus_set)


def _postprocess_stimulus_set(stimulus_set):
    from brainscore_core.supported_data_standards.brainio.stimuli import (
        StimulusSet,
    )

    df = stimulus_set.copy()
    df['image_label'] = df['label'].map({1: 'real', 0: 'pseudo'})
    df['numeric_label'] = df['label']
    df['sentence'] = df['word']
    df['image_file_name'] = [
        str(stimulus_set.stimulus_paths[stimulus_id])
        for stimulus_id in df['stimulus_id']
    ]
    df = df[[
        'stimulus_id',
        'image_file_name',
        'sentence',
        'image_label',
        'numeric_label',
        'word',
        'realpseudo',
    ]]

    processed = StimulusSet(df)
    processed.identifier = STIMULUS_SET_ID
    processed.stimulus_paths = dict(stimulus_set.stimulus_paths)
    return processed


def load_yeatman2021_assembly():
    """Load the 60k-trial ROAR human behavioral dataset from S3."""
    from brainscore_core.supported_data_standards.brainio.s3 import (
        load_assembly_from_s3,
    )

    return load_assembly_from_s3(
        identifier=ASSEMBLY_ID,
        version_id=ASSEMBLY_VERSION_ID,
        sha1=ASSEMBLY_SHA1,
        bucket=BUCKET,
        cls=_dataset_from_dataarray,
        merge_stimulus_set_meta=False,
    )


def _dataset_from_dataarray(data):
    dataset = data.to_dataset(name=data.name or 'data')
    dataset.attrs.update(data.attrs)
    return dataset


stimulus_set_registry[STIMULUS_SET_ID] = load_yeatman2021_stimulus_set
data_registry[ASSEMBLY_ID] = load_yeatman2021_assembly
