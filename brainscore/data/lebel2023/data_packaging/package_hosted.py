"""Build the hosted LeBel2023-UTS03 files from the source pickle and upload them.

    BRAINSCORE_LEBEL_PICKLE=/path/to/UTS03-assembly.pkl \\
        python -m brainscore.data.lebel2023.data_packaging.package_hosted --out /tmp/lebel [--upload]

Writes three files with the registered default settings, then prints each one's
sha1 and, with ``--upload``, its S3 version id, to paste into ``hosted.py``:

- ``assy_LeBel2023-UTS03.nc``: the responses (TR x vertex), shared by both stimulus sets.
- ``stimulus_LeBel2023-UTS03.csv``: one row per TR, the words of the preceding 5 TRs.
- ``stimulus_LeBel2023-UTS03-words.csv``: one row per spoken word, with its 32-word context.

The recordings are from OpenNeuro ds003020 (LeBel et al., 2023), released under CC0.
"""
import argparse
import hashlib
from pathlib import Path

from ..data import DEFAULT_CONTEXT_TR, DEFAULT_CONTEXT_WORDS, build, build_word_level
from ..hosted import ASSEMBLY_ID, BUCKET, FOLDER, TR_STIMULI_ID, WORD_STIMULI_ID


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tr_stimuli, assembly = build(context_window_tr=DEFAULT_CONTEXT_TR)
    word_stimuli, _, _ = build_word_level(context_words=DEFAULT_CONTEXT_WORDS)
    files = {'assembly': out / f'assy_{ASSEMBLY_ID}.nc',
             'TR stimuli': out / f'stimulus_{TR_STIMULI_ID}.csv',
             'word stimuli': out / f'stimulus_{WORD_STIMULI_ID}.csv'}
    # netCDF holds plain coordinates and scalar attrs only; the stimulus sets travel separately.
    writable = assembly.reset_index(['presentation', 'neuroid'])
    writable.attrs = {'tr_sec': assembly.attrs['tr_sec']}
    writable.to_netcdf(files['assembly'])
    tr_stimuli.to_csv(files['TR stimuli'], index=False)
    word_stimuli.to_csv(files['word stimuli'], index=False)
    return files


def upload(path: Path) -> str:
    import boto3
    key = f'{FOLDER}/{path.name}'
    boto3.client('s3').upload_file(str(path), BUCKET, key)
    return boto3.client('s3').head_object(Bucket=BUCKET, Key=key)['VersionId']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    for label, path in write(args.out).items():
        version = upload(path) if args.upload else None
        print(f'{label}: file={path.name} sha1={sha1(path)} version_id={version}')


if __name__ == '__main__':
    main()
