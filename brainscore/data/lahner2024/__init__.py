"""Data plugin for Lahner2024 BOLDMoments assets."""

from typing import Optional

from brainscore import data_registry, stimulus_set_registry


STIMULUS_ID = 'BOLDMoments'
STIMULUS_BUCKET = 'brainscore-storage/brainscore-vision/benchmarks/Lahner2024-fMRI'
STIMULUS_CSV_SHA1 = '0b27388f5898c908f58cd1f21f8f5cb3eda8536e'
STIMULUS_ZIP_SHA1 = 'dc9c3bf631632cd433d02f2f1847fd33c01ae0b3'
STIMULUS_CSV_VERSION_ID = 'WaGkWh59b1drhy1MmAVVSxh7_VT_eTay'
STIMULUS_ZIP_VERSION_ID = 'OxpOYy_3bveay9NFFFxNCVyghyAbqyIt'

ASSEMBLY_ID = 'Lahner2024-fMRI'
ASSEMBLY_VERSION_ID = 'zr_i3T9Saww44rPNJwLaxo0hgp8rYjPO'
ASSEMBLY_SHA1 = '2c7f1d2e5724b8cc3c5cf47986e956c4f13001e4'

TIMERESOLVED_ASSEMBLY_ID = 'Lahner2024-fMRI-timeresolved'
TIMERESOLVED_ASSEMBLY_VERSION_ID: Optional[str] = '6VFZOroMxaNg1P_xSQAYwGq7.Fc63zz_'
TIMERESOLVED_ASSEMBLY_SHA1: Optional[str] = '4d06589dde6dddf273a489a64da1337940d5fafd'
TIMERESOLVED_EVENTS_ID = 'Lahner2024-fMRI-timeresolved-events'
TIMERESOLVED_EVENTS_VERSION_ID: Optional[str] = 'MYar7u8KE_D.i4MXZ83GurH1EDECZ.jo'
TIMERESOLVED_EVENTS_SHA1: Optional[str] = '783c5b33a75f121812a49b8b0a7639b9ed07c6a3'
TIMERESOLVED_MOTION_ID = 'Lahner2024-fMRI-timeresolved-motion'
TIMERESOLVED_MOTION_VERSION_ID: Optional[str] = 'wR1CwATaRjpEgvpR4ozr_UzkTS6yitjR'
TIMERESOLVED_MOTION_SHA1: Optional[str] = '827130c78f6b706b52f6d0d28962eb2301bee00b'

MOTION_COLUMNS = [
    'trans_x', 'trans_y', 'trans_z',
    'rot_x', 'rot_y', 'rot_z',
    'framewise_displacement',
    'csf', 'white_matter',
]


def _require_hosted(name: str, version_id: Optional[str], sha1: Optional[str],
                    prep_script: str) -> None:
    if version_id is None or sha1 is None:
        raise RuntimeError(
            f"{name} is not yet hosted on S3. Run {prep_script} on EC2 and "
            "update unified/brainscore/data/lahner2024/__init__.py with the "
            "resulting version and sha1."
        )


def load_lahner2024_stimulus_set():
    from brainscore_core.supported_data_standards.brainio.s3 import (
        load_stimulus_set_from_s3,
    )

    return load_stimulus_set_from_s3(
        identifier=STIMULUS_ID,
        bucket=STIMULUS_BUCKET,
        csv_sha1=STIMULUS_CSV_SHA1,
        zip_sha1=STIMULUS_ZIP_SHA1,
        csv_version_id=STIMULUS_CSV_VERSION_ID,
        zip_version_id=STIMULUS_ZIP_VERSION_ID,
    )


def load_lahner2024_assembly(merge_stimulus_set_meta: bool = True):
    from brainscore import load_stimulus_set
    from brainscore_core.supported_data_standards.brainio.assemblies import (
        NeuronRecordingAssembly,
    )
    from brainscore_core.supported_data_standards.brainio.s3 import (
        load_assembly_from_s3,
    )

    return load_assembly_from_s3(
        identifier=ASSEMBLY_ID,
        version_id=ASSEMBLY_VERSION_ID,
        sha1=ASSEMBLY_SHA1,
        bucket=STIMULUS_BUCKET,
        cls=NeuronRecordingAssembly,
        stimulus_set_loader=lambda: load_stimulus_set(STIMULUS_ID),
        merge_stimulus_set_meta=merge_stimulus_set_meta,
    )


def load_lahner2024_timeresolved_assembly(
        merge_stimulus_set_meta: bool = False):
    from brainscore import load_stimulus_set
    from brainscore_core.supported_data_standards.brainio.assemblies import (
        NeuronRecordingAssembly,
    )
    from brainscore_core.supported_data_standards.brainio.s3 import (
        load_assembly_from_s3,
    )

    _require_hosted(
        TIMERESOLVED_ASSEMBLY_ID,
        TIMERESOLVED_ASSEMBLY_VERSION_ID,
        TIMERESOLVED_ASSEMBLY_SHA1,
        "python -m brainscore.data.lahner2024.prepare_timeresolved_assembly",
    )
    return load_assembly_from_s3(
        identifier=TIMERESOLVED_ASSEMBLY_ID,
        version_id=TIMERESOLVED_ASSEMBLY_VERSION_ID,
        sha1=TIMERESOLVED_ASSEMBLY_SHA1,
        bucket=STIMULUS_BUCKET,
        cls=NeuronRecordingAssembly,
        stimulus_set_loader=lambda: load_stimulus_set(STIMULUS_ID),
        merge_stimulus_set_meta=merge_stimulus_set_meta,
    )


def _s3_key(filename: str) -> tuple[str, str]:
    bucket, *prefix = STIMULUS_BUCKET.split('/', 1)
    key = f'{prefix[0]}/{filename}' if prefix else filename
    return bucket, key


def _load_csv_sidecar(filename: str, version_id: Optional[str],
                      sha1: Optional[str], prep_script: str):
    import io

    import boto3
    import pandas as pd

    _require_hosted(filename, version_id, sha1, prep_script)
    bucket, key = _s3_key(filename)
    obj = boto3.client('s3').get_object(
        Bucket=bucket, Key=key, VersionId=version_id)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))


def load_lahner2024_timeresolved_events():
    return _load_csv_sidecar(
        'Lahner2024-fMRI-timeresolved-events.csv',
        TIMERESOLVED_EVENTS_VERSION_ID,
        TIMERESOLVED_EVENTS_SHA1,
        "python -m brainscore.data.lahner2024.prepare_timeresolved_assembly",
    )


def load_lahner2024_timeresolved_motion():
    return _load_csv_sidecar(
        'Lahner2024-fMRI-timeresolved-motion.csv',
        TIMERESOLVED_MOTION_VERSION_ID,
        TIMERESOLVED_MOTION_SHA1,
        "python -m brainscore.data.lahner2024.prepare_motion_sidecar",
    )


stimulus_set_registry[STIMULUS_ID] = load_lahner2024_stimulus_set
data_registry[ASSEMBLY_ID] = load_lahner2024_assembly
data_registry[TIMERESOLVED_ASSEMBLY_ID] = load_lahner2024_timeresolved_assembly
data_registry[TIMERESOLVED_EVENTS_ID] = load_lahner2024_timeresolved_events
data_registry[TIMERESOLVED_MOTION_ID] = load_lahner2024_timeresolved_motion
