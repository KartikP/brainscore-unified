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
else ``~/brainscore-data/sources/<name>``. The output is ``output``, else the place where the
plugin's benchmarks read the data (``reads``). Nothing is ever uploaded.
"""
from dataclasses import dataclass, field
import fnmatch
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
from brainscore.data import local
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
    :param writes: the files written
    :param reads: where the benchmarks read the output, the default output: the name of a local
        asset (its folder), ``'brainio'`` (each file with a registered sha1 at
        ``$BRAINIO_HOME/<sha1>/<file name>``, where the loader reads it in place of the download),
        or empty: no local place (``$BRAINSCORE_DATA_HOME/packaged/<name>``; ``writes`` says how
        the files are used)
    :param then: the one step that remains after the data is where the benchmarks read it, if any
    :param build: the plugin's ``data_packaging`` script as a command line
        (``python -m <module> <arguments>``, optionally led by ``NAME=value`` words);
        ``{source}``, ``{output}`` and ``{work}`` (a private folder for downloads and
        intermediate files, deleted after the build) are filled in. A script reads the
        source and never changes or deletes a file in it. Empty when there is no builder.
    :param sha1: file name to the sha1 the plugin registers for it
    :param cannot: when no builder exists yet: what is known and what a maintainer needs
    :param requires: glob patterns under the source folder that must each match a file or
        folder (`` | `` separates alternatives); a source that lacks one gets the missing-source
        error. A source that is one file is matched by its name.
    :param fetches: the script fetches its public source itself; no source folder is needed,
        and one that exists is only read
    """

    source: str
    access: str
    layout: str
    needs: str
    writes: str
    build: str = ''
    reads: str = ''
    then: str = ''
    sha1: dict = field(default_factory=dict)
    cannot: str = ''
    requires: tuple = ()
    fetches: bool = False

    def readme(self, name: str) -> str:
        """The ``### <name>`` block of the plugin README's ``## Rebuild`` section."""
        if self.build:
            command = f'`python -m brainscore.data package {name}`'
            if self.fetches:
                command += f'. Source folder: {self.layout}'
            else:
                command += (f' with the source in `~/brainscore-data/sources/{name}` '
                            f'(or `--source <folder>`): {self.layout}')
            command = command.rstrip('.') + f'. Output (or `--output <new folder>`): {self._place_text(name)}.'
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
            f'- Writes: {self.writes}' + (f' Then: {self.then}' if self.then else ''),
            f'- Check: {check}',
        ])

    def _place_text(self, name: str) -> str:
        if self.reads == 'brainio':
            return ('each identical file at `$BRAINIO_HOME/<sha1>/<file name>` (`~/.brainio` by default), '
                    f'where the loader reads it; the build record in `$BRAINIO_HOME/packaged/{name}`')
        if self.reads:
            asset = local.REGISTRY[self.reads]
            return f'the folder the benchmarks read, `~/{asset.default_path}` (or `${asset.env_var}`)'
        return f'`~/brainscore-data/packaged/{name}`'


def data_home() -> Path:
    """The root of the source and packaged folders."""
    return Path(os.environ.get(HOME_VARIABLE) or '~/brainscore-data').expanduser()


def source_folder(name: str) -> Path:
    """Where the source of ``name`` is expected when no folder is given (the same rule for
    every plugin)."""
    return data_home() / 'sources' / name


def _builder(name: str) -> DataBuilder:
    try:
        factory = data_builder_registry[name]
    except KeyError:
        raise ValueError(f'no data plugin packages {name!r}; known: {sorted(data_builder_registry)}') from None
    return factory()


def _missing(name: str, builder: DataBuilder, source: Path, problem: str = '') -> str:
    where = ([f'The source of {name} at', f'    {source}', f'is incomplete: {problem}'] if problem
             else [f'The source of {name} is not at', f'    {source}'])
    return '\n'.join([
        *where, '',
        f'Data needed: {builder.source}',
        f'Access: {builder.access}',
        f'Put it in that folder, laid out as: {builder.layout}',
        f'Needs: {builder.needs}', '',
        f'Then run again:  python -m brainscore.data package {name}',
        f'(or give another folder with --source, or set {HOME_VARIABLE} so that '
        f'{HOME_VARIABLE}/sources/{name} is the folder)',
    ])


def _lacks(source: Path, patterns: tuple) -> str:
    """The first required pattern that matches nothing in ``source``, or ''."""
    for pattern in patterns:
        alternatives = [alternative.strip() for alternative in pattern.split('|')]
        if source.is_file():
            found = any(fnmatch.fnmatch(source.name, alternative.rsplit('/', 1)[-1]) for alternative in alternatives)
        else:
            found = any(next(source.glob(alternative), None) is not None for alternative in alternatives)
        if not found:
            return f'nothing matches {pattern}'
    return ''


def _run(command: str, source: Path, output: Path, work: Path) -> None:
    """Run a ``python -m`` command line in this process, as the plugin's script."""
    words = shlex.split(command.format(**{key: shlex.quote(str(value)) for key, value in
                                          (('source', source), ('output', output), ('work', work))}))
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


_TOLERANCE = 1e-6      # float parts agree when within this fraction of the sum of magnitudes (at least 1)


def _digest(values) -> str:
    """The sha1 of values compared exactly (text, integers, booleans, coordinates), in order."""
    digest = hashlib.sha1()
    for value in values:
        digest.update(str(value).encode() + b'\x1f')
    return digest.hexdigest()


def _floats(values) -> dict:
    """Float values as counts of missing values and three sums; the weighted sum sees order."""
    import numpy as np
    values = np.asarray(values, dtype=float).ravel()
    missing = np.isnan(values)
    values = np.where(missing, 0.0, values)
    weights = np.arange(1, values.size + 1) / max(values.size, 1)
    return {'missing': int(missing.sum()), 'sum': float(values.sum()),
            'weighted': float((values * weights).sum()), 'magnitude': float(np.abs(values).sum())}


def _values(values) -> dict:
    import numpy as np
    values = np.asarray(values)
    return _floats(values) if values.dtype.kind == 'f' else {'sha1': _digest(values.ravel())}


def _content(path: Path) -> dict:
    """A small summary of a file's content, by its kind: the unit of the content check.

    CSV: rows, columns, each column's values. JSON: list items, every leaf with its
    place. netCDF: dimensions, coordinates, data variables. Any other kind: its sha1.
    Floats are kept as sums (compared with a tolerance); all else as a sha1 (compared exactly).
    """
    if path.suffix == '.csv':
        import pandas as pd
        table = pd.read_csv(path)
        return {'kind': 'csv', 'rows': len(table), 'columns': list(map(str, table.columns)),
                'values': {str(column): _values(table[column].to_numpy()) for column in table.columns}}
    if path.suffix == '.json':
        content = json.loads(path.read_text())
        exact, floats = [], []

        def walk(node, place):
            if isinstance(node, dict):
                for key in sorted(node):
                    walk(node[key], f'{place}/{key}')
            elif isinstance(node, list):
                for index, item in enumerate(node):
                    walk(item, f'{place}/{index}')
            elif isinstance(node, float):
                floats.append(node)
            else:
                exact.append(f'{place}={json.dumps(node)}')
        walk(content, '')
        return {'kind': 'json', 'items': len(content) if isinstance(content, list) else None,
                'leaves': _digest(exact), 'floats': _floats(floats)}
    if path.suffix == '.nc':
        import xarray as xr
        with xr.open_dataset(path) as data:
            return {'kind': 'netcdf',
                    'dimensions': {str(dim): int(size) for dim, size in data.sizes.items()},
                    'coordinates': {str(name): {'dimensions': list(map(str, coord.dims)),
                                                'sha1': _digest(coord.values.astype(str).ravel())}
                                    for name, coord in data.coords.items()},
                    'variables': {str(name): {'dimensions': list(map(str, variable.dims)), **_values(variable.values)}
                                  for name, variable in data.data_vars.items()}}
    return {'kind': 'bytes', 'sha1': _sha1(path)}


def _counts(content: dict) -> str:
    """The counts a person reads: rows of a CSV, dimension sizes of a netCDF file, items of a JSON list."""
    if content['kind'] == 'csv':
        return f"rows {content['rows']}"
    if content['kind'] == 'netcdf':
        return ', '.join(f'{dim} {size}' for dim, size in content['dimensions'].items())
    if content['kind'] == 'json' and content['items'] is not None:
        return f"items {content['items']}"
    return ''


def _differences(built, expected, place: str = '') -> list:
    """What differs between two content summaries: exact parts equal, float parts within the tolerance."""
    if isinstance(expected, dict) and 'magnitude' in expected and isinstance(built, dict) and 'magnitude' in built:
        found = []
        if built['missing'] != expected['missing']:
            found.append(f"{place}: missing values {built['missing']} != {expected['missing']}")
        tolerance = _TOLERANCE * max(1.0, expected['magnitude'])
        largest = max(abs(built[key] - expected[key]) for key in ('sum', 'weighted', 'magnitude'))
        if largest > tolerance:
            found.append(f'{place}: values differ by up to {largest:.6g} (tolerance {tolerance:.3g})')
        return found
    if isinstance(expected, dict) and isinstance(built, dict):
        found = []
        for key in [*expected, *(key for key in built if key not in expected)]:
            where = f'{place} {key}'.strip()
            if key not in built:
                found.append(f'{where}: absent')
            elif key not in expected:
                found.append(f'{where}: not expected')
            else:
                found += _differences(built[key], expected[key], where)
        return found
    return [] if built == expected else [f'{place}: {built} != {expected}']


def brainio_home() -> Path:
    """Where brainio keeps hosted files: ``$BRAINIO_HOME/<sha1>/<file name>``."""
    return Path(os.environ.get('BRAINIO_HOME') or '~/.brainio').expanduser()


def _packaging_folder(builder: DataBuilder) -> Path:
    """The plugin's ``data_packaging`` folder: the folder of the build script."""
    import importlib.util
    words = shlex.split(builder.build)
    return Path(importlib.util.find_spec(words[words.index('-m') + 1]).origin).parent


def _expected(name: str, builder: DataBuilder) -> dict:
    """``{file path: entry of build.json}`` that a cloud run recorded in ``data_packaging/expected.json``."""
    path = _packaging_folder(builder) / 'expected.json'
    return json.loads(path.read_text()).get(name, {}) if path.exists() else {}


def _registered_copy(builder: DataBuilder, file: str, sha1: str) -> Path | None:
    """The registered file on this machine, found by its sha1: the brainio cache or the plugin folder."""
    for candidate in (brainio_home() / sha1 / file, _packaging_folder(builder).parent / file):
        if candidate.is_file() and _sha1(candidate) == sha1:
            return candidate
    return None


def _check(builder: DataBuilder, recorded: dict, relative: str, entry: dict) -> tuple:
    """``(outcome, problem)`` of one written file against its expected record; ``problem`` is '' or why it fails."""
    file = Path(relative).name
    sha1 = builder.sha1.get(file) or recorded.get('sha1')
    if not sha1:
        return f"no expected record: {_counts(entry['content']) or str(entry['bytes']) + ' bytes'}", ''
    if sha1 == entry['sha1']:
        return 'identical', ''
    copy = _registered_copy(builder, file, sha1)
    expected = _content(copy) if copy else recorded.get('content')
    if expected is None:
        return 'failed', (f'its sha1 differs from the expected {sha1}, and no registered copy is on this machine '
                          f'to compare the content: put the registered file at {brainio_home() / sha1 / file} '
                          '(`brainscore.load_dataset` fetches a hosted file there), then run again with --output')
    differences = _differences(entry['content'], expected)
    if differences:
        return 'failed', 'its content differs from the expected record: ' + '; '.join(differences[:5]) + (
            f'; and {len(differences) - 5} more' if len(differences) > 5 else '')
    return 'same content, bytes differ', ''


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
    if not builder.fetches:
        if not native.exists():
            raise LocalDataMissing(_missing(name, builder, native))
        lacks = _lacks(native, builder.requires)
        if lacks:
            raise LocalDataMissing(_missing(name, builder, native, lacks))
    read_place = _read_place(name, builder).absolute()
    destination = Path(output).expanduser().absolute() if output is not None else read_place
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'{destination} already exists; give a new output folder')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Build beside the destination and rename only after success: an interrupted
    # build must not look like a ready dataset.
    staging = Path(tempfile.mkdtemp(prefix='.brainscore-build-', dir=destination.parent))
    work = Path(tempfile.mkdtemp(prefix='.brainscore-work-', dir=destination.parent))
    try:
        try:
            _run(builder.build, native, staging, work)
        except FileNotFoundError as missing:
            if builder.fetches:
                raise
            # A required input the script did not find: the source is incomplete.
            raise LocalDataMissing(_missing(name, builder, native, str(missing))) from missing
        files = {}
        for path in sorted(p for p in staging.rglob('*') if p.is_file()):
            if path.is_symlink():
                raise ValueError('a data script must not create symlinks')
            relative = path.relative_to(staging).as_posix()
            files[relative] = {'bytes': path.stat().st_size, 'sha1': _sha1(path), 'content': _content(path)}
        recorded = _expected(name, builder)
        problems = []
        for relative, entry in files.items():
            entry['check'], problem = _check(builder, recorded.get(relative, {}), relative, entry)
            if problem:
                problems.append(f'  {relative}: {problem}')
        if problems:
            raise ValueError('\n'.join([f'The check of the built {name} files failed; nothing was moved '
                                         f'to {destination}.', *problems]))
        placed = _place_hosted(builder, staging, files) if builder.reads == 'brainio' and destination == read_place else {}
        (staging / 'build.json').write_text(json.dumps(
            {'schema_version': 3, 'name': name, 'source': builder.source, 'files': files},
            indent=2, allow_nan=False) + '\n')
        # Keep packaged data private by default: some sources forbid redistribution.
        for entry in [staging, *staging.rglob('*')]:
            entry.chmod(0o700 if entry.is_dir() else 0o600)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('the output appeared during the build; nothing replaced')
        os.rename(staging, destination)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if staging.exists():
            shutil.rmtree(staging)
    _report(name, builder, destination, read_place, files, placed)
    return destination


def _read_place(name: str, builder: DataBuilder) -> Path:
    """Where the plugin's benchmarks read the output: the default output folder."""
    if builder.reads == 'brainio':
        return brainio_home() / 'packaged' / name
    if builder.reads:
        return local.REGISTRY[builder.reads].resolved()
    return data_home() / 'packaged' / name


def _place_hosted(builder: DataBuilder, staging: Path, files: dict) -> dict:
    """Move each identical hosted file to ``$BRAINIO_HOME/<sha1>/<file name>``; ``{file path: place}``.

    A file already there is not replaced. The staging folder is under ``$BRAINIO_HOME``, so each
    move is one rename on one volume.
    """
    placed = {}
    for relative, entry in files.items():
        file = Path(relative).name
        if entry['check'] != 'identical' or file not in builder.sha1:
            continue
        place = brainio_home() / entry['sha1'] / file
        if place.exists():
            continue
        place.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging / relative, place)
        placed[relative] = place
    return placed


def _report(name: str, builder: DataBuilder, destination: Path, read_place: Path,
            files: dict, placed: dict) -> None:
    print(f'Packaged {name}.')
    for index, (relative, entry) in enumerate(files.items()):
        if index == _LISTED_FILES:
            print(f'  ... and {len(files) - index} more files (all in build.json)')
            break
        counts = _counts(entry['content'])
        print(f"  {relative}  {entry['bytes']} bytes  sha1 {entry['sha1']}" + (f'  ({counts})' if counts else ''))
        print(f"    check: {entry['check']}")
    # (i) where the data is now, (ii) whether the benchmarks read it there, (iii) the check.
    if placed:
        print('The data is now in:')
        for place in placed.values():
            print(f'    {place}')
        print(f'    {destination} (build.json' + (', and the files that are not placed)' if len(placed) < len(files) else ')'))
    else:
        print(f'The data is now in: {destination}')
    if destination != read_place:
        if builder.reads == 'brainio':
            step = (f'The loader reads a hosted file only at {brainio_home()}/<sha1>/<file name>; one step '
                    f'remains: run again without --output to place the identical files there.')
        elif builder.reads:
            step = (f'The benchmarks read {read_place}; one step remains: set '
                    f'{local.REGISTRY[builder.reads].env_var}={destination}.')
        else:
            step = builder.writes
    elif builder.reads == 'brainio':
        kept = [relative for relative in files if Path(relative).name in builder.sha1 and relative not in placed]
        unread = [relative for relative in files if Path(relative).name not in builder.sha1]
        step = 'The loader reads each placed file there in place of the download.' if placed else ''
        if kept:
            step += (f' The loader accepts only the registered bytes, so for {", ".join(kept)} the benchmarks '
                     'read the hosted file (already on this machine, or downloaded on first use).')
        if unread:
            step += f' {", ".join(unread)}: no registered sha1, so no loader reads it.'
        step = step.strip()
    elif builder.reads:
        step = 'The benchmarks read it there.'
    else:
        step = builder.writes
    print(step + (f' One step remains: {builder.then}' if builder.then else
                  ' No more steps.' if builder.reads and destination == read_place else ''))
    outcomes = {}
    for entry in files.values():
        outcome = 'no expected record' if entry['check'].startswith('no expected record') else entry['check']
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    print('Check: ' + ', '.join(f'{count} {outcome}' for outcome, count in outcomes.items()) + '.')


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
