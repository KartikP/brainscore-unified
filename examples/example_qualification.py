"""Qualification helpers shared by the CogGym and native-environment examples.

They check a staged model snapshot against a pinned Hub listing, record the
software environment, write a report on success and on failure, and check that
an ablation zeroed only the selected units. Nothing here loads a model.
"""
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from brainscore.run_record import RunRecord


def require(condition: bool, message: str) -> None:
    """Check one qualification gate. Unlike assert, this also runs under python -O."""
    if not condition:
        raise RuntimeError(f'Qualification check failed: {message}')


def snapshot_digest(directory: Path) -> str:
    """Hash file names and contents of a local model snapshot (follows HF cache links)."""
    if not directory.is_dir():
        raise FileNotFoundError(f'Stage the model snapshot locally first: {directory}')
    digest = hashlib.sha256()
    count = 0
    for path in sorted(directory.rglob('*')):
        if not path.is_file():
            continue
        with path.open('rb') as stream:
            content = hashlib.file_digest(stream, 'sha256').digest()
        digest.update(path.relative_to(directory).as_posix().encode() + b'\0')
        digest.update(content)
        count += 1
    if count == 0:
        raise ValueError(f'Model snapshot is empty: {directory}')
    return digest.hexdigest()


# Hub listing of the loader's files at the pinned commit (metadata only). sha256 is the
# LFS hash of the file; git_sha1 is the Git blob hash Git uses for small files.
SMOLVLM_MANIFEST = {
    'repository': 'HuggingFaceTB/SmolVLM-256M-Instruct',
    'revision': '7e3e67edbbed1bf9888184d9df282b700a323964',
    'files': {
        'added_tokens.json': {'size': 4739, 'git_sha1': '71dc811700b6aa9274d3697ee7d4fb2d5375e766'},
        'chat_template.json': {'size': 429, 'git_sha1': '214a30563adf9598b26cdd3ad3bbafca9d7af45d'},
        'config.json': {'size': 7353, 'git_sha1': '1c09d62975f946e7e07f62ec4db3f16c9c701144'},
        'generation_config.json': {'size': 136, 'git_sha1': '14051e31eab5938a09976bf8c3e4769491f53a80'},
        'merges.txt': {'size': 466391, 'git_sha1': '69503b13f727ba3812b6803e97442a6de05ef5eb'},
        'model.safetensors': {
            'size': 513028808,
            'sha256': '74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e'},
        'preprocessor_config.json': {'size': 486, 'git_sha1': '893236748032d4bf033823d6a4a0d62b78b6bc0e'},
        'processor_config.json': {'size': 68, 'git_sha1': 'b3b1deadd0e6e18c2070fdb0c29074892ea58ffe'},
        'special_tokens_map.json': {'size': 1069, 'git_sha1': '647565708c361dc42350b5c7b79c46a4ff153b18'},
        'tokenizer.json': {'size': 3548256, 'git_sha1': 'a4005d1cf3170a31600a5c96f95768166cbc2b28'},
        'tokenizer_config.json': {'size': 28249, 'git_sha1': 'c85c07d72e6ffca6b6ff138a8e8f09ae65740bfb'},
        'vocab.json': {'size': 800662, 'git_sha1': '0ad5ecc2035b7031b88afb544ee95e2d49baa484'},
    },
    # Present in the repository but never read by the loader; not verified.
    'ignored': ['.gitattributes', 'README.md', 'onnx', '.cache'],
}


def file_hash(path: Path, kind: str) -> str:
    """sha256 of the bytes, or the Git blob SHA-1 ('blob <size>\\0' + bytes)."""
    with path.open('rb') as stream:
        if kind == 'sha256':
            return hashlib.file_digest(stream, 'sha256').hexdigest()
        digest = hashlib.sha1(b'blob %d\0' % path.stat().st_size)
        return hashlib.file_digest(stream, lambda: digest).hexdigest()


def verify_model_identity(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check a staged snapshot against a pinned Hub listing before anything loads it.

    Accepts a Hugging Face cache snapshot (.../snapshots/<commit>, files may be
    links into blobs/) or a huggingface-cli --local-dir download. Every file the
    loader reads must exist with the listed size and hash, computed from the bytes
    on disk. Any other top-level entry must be on the manifest's ignore list, so an
    extra file cannot change what is loaded. A cache snapshot name or a local-dir
    download record naming another commit fails.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f'Stage the model snapshot locally first: {directory}')
    revision = manifest['revision']
    if directory.parent.name == 'snapshots':
        require(directory.name == revision,
                f'cache snapshot {directory.name} is not the pinned revision {revision}')
    files, ignored = manifest['files'], set(manifest['ignored'])
    unexpected = sorted(p.name for p in directory.iterdir()
                        if p.name not in files and p.name not in ignored)
    require(not unexpected, f'unexpected files in the model snapshot: {unexpected}')
    records = directory / '.cache' / 'huggingface' / 'download'
    for name, expected in sorted(files.items()):
        path = directory / name
        require(path.is_file(), f'model file {name} is missing')
        require(path.stat().st_size == expected['size'], f'model file {name} has the wrong size')
        kind = 'sha256' if 'sha256' in expected else 'git_sha1'
        require(file_hash(path, kind) == expected[kind],
                f'model file {name} does not match {manifest["repository"]}@{revision}')
        record = records / f'{name}.metadata'
        if record.is_file():
            # huggingface-cli writes the commit on the record's first line.
            require(record.read_text().splitlines()[:1] == [revision],
                    f'download record for {name} names another revision')
    return {'repository': manifest['repository'], 'revision': revision,
            'verified_files': sorted(files),
            'method': 'size and content hash of each loader file against the Hub listing'}


def source_revision(directory: Path) -> dict[str, Any]:
    """Return the Git commit and tracked-edit state, or None for a non-Git directory."""
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(directory), *arguments],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return {'commit': git('rev-parse', 'HEAD'),
                'tracked_edits': bool(git('status', '--porcelain', '--untracked-files=no'))}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'tracked_edits': None}


def environment() -> dict[str, Any]:
    """Versions that can change a result; missing packages are reported as None."""
    versions = {}
    for name in ('brainscore', 'brainscore-core', 'torch', 'transformers', 'numpy',
                 'gymnasium', 'minigrid'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': versions,
            'unified_source': source_revision(Path(__file__).resolve().parent)}


@contextmanager
def qualification_report(path: Path, report: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Write the report whether the checks pass or fail; never claim a pass early."""
    report['status'] = 'failed'
    try:
        yield report
        report['status'] = 'passed'
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, default=str) + '\n')


def check_ablation(record: "RunRecord", indices: list[int], total_units: int) -> None:
    """Check that only the selected unit outputs were zeroed."""
    events = list(record.events())
    before = [
        event['payload']['value']['array'] for event in events
        if event['kind'] == 'activity' and event['payload']['when'] == 'before'
    ]
    after = [
        event['payload']['value']['array'] for event in events
        if event['kind'] == 'activity' and event['payload']['when'] == 'after'
    ]
    require(before and len(before) == len(after), 'paired before/after activity records')
    require(any((array[..., indices] != 0).any() for array in before),
            'selected units were active before the intervention')
    require(all((array[..., indices] == 0).all() for array in after),
            'selected units are zero after the intervention')
    unselected = sorted(set(range(total_units)) - set(indices))
    require(all(
        (original[..., unselected] == changed[..., unselected]).all()
        for original, changed in zip(before, after)
    ), 'unselected units are unchanged')
