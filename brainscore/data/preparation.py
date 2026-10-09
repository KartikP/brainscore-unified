"""Build author-supplied datasets locally through registered data plugins."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable


@dataclass(frozen=True)
class DataBuilder:
    """A plugin converts native files; an optional resolver obtains authorized files.

    ``build(source, destination)`` writes into a private staging directory and
    returns JSON provenance. ``resolve(request_id)`` returns a local source path
    and provenance, using the provider's normal authentication. Neither callback
    should return credentials or private access identifiers in provenance.
    """

    build: Callable[[Path, Path], dict]
    resolve: Callable[[str], tuple[Path, dict]] | None = None


# Factories are lazy: listing builders must not download data or import ML code.
data_builder_registry: dict[str, Callable[[], DataBuilder]] = {}


def prepare_dataset(
    identifier: str,
    *,
    output: str | Path,
    source: str | Path | None = None,
    request_id: str | None = None,
) -> Path:
    """Validate and build a local dataset without replacing existing data."""
    if (source is None) == (request_id is None):
        raise ValueError('Provide exactly one of source or request_id')
    try:
        factory = data_builder_registry[identifier]
    except KeyError:
        raise ValueError(f'No local data builder registered for {identifier}') from None
    destination = Path(output).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Output already exists; choose a new directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.brainscore-build-', dir=destination.parent))
    try:
        builder = factory()
        acquisition = {'method': 'local_files'}
        if request_id is not None:
            if builder.resolve is None:
                raise ValueError('This dataset has no request-ID resolver; provide source')
            source, acquisition = builder.resolve(request_id)
        native = Path(source).expanduser()
        if not native.exists():
            raise FileNotFoundError('Native source is missing; provide downloaded author files')
        provenance = builder.build(native, staging)
        (staging / 'build.json').write_text(json.dumps({
            'schema_version': 1,
            'dataset': identifier,
            'acquisition': acquisition,
            'provenance': provenance,
        }, indent=2, allow_nan=False) + '\n')
        # Keep local text and derived data private by default.
        for entry in staging.rglob('*'):
            if entry.is_symlink():
                raise ValueError('Data builders must not create symlinks')
            entry.chmod(0o700 if entry.is_dir() else 0o600)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Output appeared during build; nothing replaced')
        os.rename(staging, destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)
