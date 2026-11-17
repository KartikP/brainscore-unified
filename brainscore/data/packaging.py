"""Package a data plugin's files from the original source: ``package_data(name)``.

Each data plugin registers a factory of one :class:`DataBuilder` per packaged dataset
beside its registration (Contract Rules 11.4)::

    data_builder_registry['mystudy2026'] = lambda: DataBuilder(
        source='https://example.org/mystudy2026; licence CC-BY 4.0',
        access='public',
        layout='responses.mat and the stimuli/ folder, as the download unpacks',
        needs='scipy; 2 GB of disk; minutes',
        writes='assy_MyStudy2026.nc, stimulus_MyStudy2026.csv',
        build='python -m brainscore.data.mystudy2026.data_packaging.package --source {source} --out {output}',
        sha1={'assy_MyStudy2026.nc': ASSEMBLY_SHA1},
    )

The source folder is the ``source`` argument, else ``$BRAINSCORE_DATA_HOME/sources/<name>``,
else ``~/brainscore-data/sources/<name>``; the output folder is ``output``, else
``.../packaged/<name>`` under the same root. Nothing is ever uploaded.
"""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import sys
import tempfile
import warnings

from brainscore import data_builder_registry
from brainscore.data.local import LocalDataMissing

HOME_VARIABLE = 'BRAINSCORE_DATA_HOME'
_LISTED_FILES = 20      # files printed one by one; the rest are only in build.json


@dataclass(frozen=True)
class DataBuilder:
    """The facts of one packaged dataset, and the command that packages it.

    :param source: what data is needed and from where: the address and the licence, as the source states them
    :param access: what access it needs: public, an account, a data-use agreement or private, and how to ask
    :param layout: the files ``package_data`` expects in the source folder
    :param needs: the tools, the disk space and the time
    :param writes: the files written, and how they are used in place of a download
    :param build: the plugin's ``data_packaging`` script as a command line
        (``python -m <module> <arguments>``, optionally led by ``NAME=value`` words);
        ``{source}`` and ``{output}`` are filled in. Empty when there is no builder.
    :param sha1: file name to the sha1 the plugin registers for it
    :param cannot: when no builder exists yet: what is known and what a maintainer needs
    :param fetches: the script fetches its public source itself; no source folder is needed
    """

    source: str
    access: str
    layout: str
    needs: str
    writes: str
    build: str = ''
    sha1: dict = field(default_factory=dict)
    cannot: str = ''
    fetches: bool = False

    def readme(self, name: str) -> str:
        """The ``### <name>`` block of the plugin README's ``## Rebuild`` section."""
        if self.build:
            command = f'`python -m brainscore.data package {name}`'
            if not self.fetches:
                command += (f' with the source in `~/brainscore-data/sources/{name}` '
                            f'(or `--source <folder>`): {self.layout}')
        elif self.cannot:
            command = f'none yet. {self.cannot}'
        else:
            command = 'none: nothing to package.'
        check = ('; '.join(f'sha1 of `{file}` is `{value}`' for file, value in self.sha1.items())
                 or ('the printed counts; no sha1 is recorded for these files' if self.build
                     else 'nothing to compare.'))
        return '\n'.join([
            f'### {name}', '',
            f'- Source: {self.source}',
            f'- Access: {self.access}',
            f'- Needs: {self.needs}',
            f'- Command: {command}',
            f'- Writes: {self.writes}',
            f'- Check: {check}',
        ])


def data_home() -> Path:
    """The root of the source and packaged folders."""
    return Path(os.environ.get(HOME_VARIABLE) or '~/brainscore-data').expanduser()


def source_folder(name: str) -> Path:
    """Where the source of ``name`` is expected when no folder is given."""
    return data_home() / 'sources' / name


def _builder(name: str) -> DataBuilder:
    try:
        factory = data_builder_registry[name]
    except KeyError:
        raise ValueError(f'no data plugin packages {name!r}; known: {sorted(data_builder_registry)}') from None
    return factory()


def _missing(name: str, builder: DataBuilder, source: Path) -> str:
    return '\n'.join([
        f'The source of {name} is not at', f'    {source}', '',
        f'Data needed: {builder.source}',
        f'Access: {builder.access}',
        f'Put it in that folder, laid out as: {builder.layout}',
        f'Needs: {builder.needs}', '',
        f'Then run again:  python -m brainscore.data package {name}',
        f'(or give another folder with --source, or set {HOME_VARIABLE} so that '
        f'{HOME_VARIABLE}/sources/{name} is the folder)',
    ])


def _run(command: str, source: Path, output: Path) -> None:
    """Run a ``python -m`` command line in this process, as the plugin's script."""
    words = shlex.split(command.format(source=shlex.quote(str(source)), output=shlex.quote(str(output))))
    environment = {}
    while words and '=' in words[0]:
        key, value = words.pop(0).split('=', 1)
        environment[key] = value
    if words[:2] != ['python', '-m'] or len(words) < 3:
        raise ValueError(f'a build command is "python -m <module> ...", not {command!r}')
    before = {key: os.environ.get(key) for key in environment}
    argv = sys.argv
    os.environ.update(environment)
    sys.argv = words[2:]
    try:
        with warnings.catch_warnings():
            # The script's module may be imported already (by a test or a plugin); that is harmless.
            warnings.filterwarnings('ignore', message='.*found in sys.modules after import', category=RuntimeWarning)
            runpy.run_module(words[2], run_name='__main__', alter_sys=True)
    except SystemExit as stop:
        if stop.code not in (None, 0):
            raise RuntimeError(f'{words[2]} stopped: {stop.code}') from None
    finally:
        sys.argv = argv
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _counts(path: Path) -> dict:
    """Rows of a CSV, dimension sizes of a netCDF file, items of a JSON list."""
    if path.suffix == '.csv':
        import pandas as pd
        return {'rows': len(pd.read_csv(path, dtype=str, keep_default_na=False))}
    if path.suffix == '.nc':
        import xarray as xr
        with xr.open_dataset(path) as data:
            return {str(dim): int(size) for dim, size in data.sizes.items()}
    if path.suffix == '.json':
        content = json.loads(path.read_text())
        return {'items': len(content)} if isinstance(content, list) else {}
    return {}


def package_data(name: str, source: str | Path | None = None,
                 output: str | Path | None = None) -> Path | None:
    """Package the files of data plugin ``name`` from its original source.

    Returns the output folder, or ``None`` when there is nothing to package.
    Raises ``LocalDataMissing`` with what to get and where to put it when the source
    is missing, and ``NotImplementedError`` when no builder exists yet.
    """
    builder = _builder(name)
    if builder.cannot:
        raise NotImplementedError(
            f'{name} cannot be packaged from its source yet. {builder.cannot}\n'
            f'Known source: {builder.source}\nAccess: {builder.access}')
    if not builder.build:
        print(f'{name}: nothing to package. {builder.writes}')
        return None
    native = Path(source).expanduser() if source is not None else source_folder(name)
    if builder.fetches:
        native.mkdir(parents=True, exist_ok=True)
    elif not native.exists():
        raise LocalDataMissing(_missing(name, builder, native))
    destination = (Path(output).expanduser() if output is not None
                   else data_home() / 'packaged' / name).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'{destination} already exists; give a new output folder')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Build beside the destination and rename only after success: an interrupted
    # build must not look like a ready dataset.
    staging = Path(tempfile.mkdtemp(prefix='.brainscore-build-', dir=destination.parent))
    try:
        _run(builder.build, native, staging)
        files = {}
        for path in sorted(p for p in staging.rglob('*') if p.is_file()):
            if path.is_symlink():
                raise ValueError('a data script must not create symlinks')
            relative = path.relative_to(staging).as_posix()
            files[relative] = {'bytes': path.stat().st_size, 'sha1': _sha1(path), 'counts': _counts(path)}
        (staging / 'build.json').write_text(json.dumps(
            {'schema_version': 2, 'name': name, 'source': builder.source, 'files': files},
            indent=2, allow_nan=False) + '\n')
        # Keep packaged data private by default: some sources forbid redistribution.
        for entry in [staging, *staging.rglob('*')]:
            entry.chmod(0o700 if entry.is_dir() else 0o600)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('the output appeared during the build; nothing replaced')
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    _report(name, builder, destination, files)
    return destination


def _report(name: str, builder: DataBuilder, destination: Path, files: dict) -> None:
    print(f'Packaged {name} into {destination}')
    for index, (relative, entry) in enumerate(files.items()):
        if index == _LISTED_FILES:
            print(f'  ... and {len(files) - index} more files (all in build.json)')
            break
        counts = ', '.join(f'{key} {value}' for key, value in entry['counts'].items())
        print(f"  {relative}  {entry['bytes']} bytes  sha1 {entry['sha1']}" + (f'  ({counts})' if counts else ''))
        registered = builder.sha1.get(Path(relative).name)
        if registered is None:
            continue
        if registered == entry['sha1']:
            print(f'    equals the registered sha1. To use it in place of the download, put it at '
                  f'$BRAINIO_HOME/{registered}/{Path(relative).name} (BRAINIO_HOME defaults to ~/.brainio)')
        else:
            print(f'    differs from the registered sha1 {registered}: brainio refuses it in place of '
                  'the download. Compare its values with the hosted file before you use it.')
    print(builder.writes)


def listing() -> str:
    """Each packaged dataset, where its source is expected and whether it is there."""
    lines = []
    for name in sorted(data_builder_registry):
        builder = _builder(name)
        if builder.cannot:
            where = 'cannot be packaged yet'
        elif not builder.build:
            where = 'nothing to package'
        elif builder.fetches:
            where = 'fetches its public source itself'
        else:
            folder = source_folder(name)
            where = f"{'found' if folder.exists() else 'missing'}: {folder}"
        lines.append(f'{name:28s} {where}')
    lines.append('')
    lines.append('Package one:  python -m brainscore.data package <name>')
    return '\n'.join(lines)
