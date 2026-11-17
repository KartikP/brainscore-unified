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
from brainscore.data.local import HOME_VARIABLE, LocalDataMissing, data_home

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
        (``python -m <module> <arguments>``);
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
        recorded = _expected(name, self) if self.build else {}
        check = ('; '.join(f'sha1 of `{file}` is `{value}`' for file, value in self.sha1.items())
                 or (f'the sha1 and exact content digest of {len(recorded)} files in `data_packaging/expected.json`; '
                     'the other files: the printed counts' if recorded else '')
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
            return f'the folder the benchmarks read, `$BRAINSCORE_DATA_HOME/{self.reads}` (`~/brainscore-data/{self.reads}` by default)'
        return f'`~/brainscore-data/packaged/{name}`'


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
    if words[:2] != ['python', '-m'] or len(words) < 3:
        raise ValueError(f'a build command is "python -m <module> ...", not {command!r}')
    argv = sys.argv
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


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


TOLERANCE = 1e-6       # a finite float agrees when |built - expected| <= TOLERANCE * max(1, |expected|)
NO_PARSER = object()   # what _read gives for a kind compared by bytes only (JSON null is content)


def _read(path: Path):
    """The parsed content of a file, or ``NO_PARSER`` for a kind compared by bytes only.

    A table is read as the LeBel loader reads it (``keep_default_na=False``: text such as ``NA``
    stays text); a netCDF file as xarray loads it; JSON as its structure.
    """
    if path.suffix == '.csv':
        import pandas as pd
        return pd.read_csv(path, keep_default_na=False)
    if path.suffix == '.json':
        return json.loads(path.read_text())
    if path.suffix == '.nc':
        import xarray as xr
        with xr.open_dataset(path) as data:
            return data.load()
    return NO_PARSER


def _counts(content) -> str:
    """The counts a person reads: rows of a table, dimension sizes of a netCDF file, items of a JSON list."""
    if hasattr(content, 'columns'):
        return f'rows {len(content)}'
    if hasattr(content, 'data_vars'):
        return ', '.join(f'{dim} {size}' for dim, size in content.sizes.items())
    return f'items {len(content)}' if isinstance(content, list) else ''


def _array_differences(built, expected, where: str, exact: bool = False) -> list:
    """Type kind and shape exactly; missing values at the same places; each float within the tolerance
    of its own expected value (``exact``: equal); every other value equal."""
    import numpy as np
    built, expected = np.asarray(built), np.asarray(expected)
    if built.dtype.kind != expected.dtype.kind or built.shape != expected.shape:
        return [f'{where}: type {built.dtype.kind} shape {built.shape} != type {expected.dtype.kind} '
                f'shape {expected.shape}']
    if built.dtype.kind != 'f':
        differ = int(np.sum(built != expected))
        return [f'{where}: {differ} values differ'] if differ else []
    missing = np.isnan(built)
    if (missing != np.isnan(expected)).any():
        return [f'{where}: missing values at other places']
    with np.errstate(invalid='ignore'):
        gap = np.abs(built - expected)
        finite = np.isfinite(built) & np.isfinite(expected)      # infinities must be equal exactly
        off = ~((built == expected) | missing
                | (not exact and finite & (gap <= TOLERANCE * np.maximum(1, np.abs(expected)))))
    return [f'{where}: {int(off.sum())} values differ, the largest by {np.nanmax(gap[off]):.6g}'] if off.any() else []


def _table_differences(built, expected) -> list:
    if list(built.columns) != list(expected.columns) or len(built) != len(expected):
        return [f'columns {list(built.columns)}, rows {len(built)} != columns {list(expected.columns)}, '
                f'rows {len(expected)}']
    return [found for column in expected.columns for found in
            _array_differences(built[column].to_numpy(), expected[column].to_numpy(), f'column {column}')]


def _netcdf_differences(built, expected) -> list:
    if dict(built.sizes) != dict(expected.sizes):
        return [f'dimensions {dict(built.sizes)} != {dict(expected.sizes)}']
    found = []
    for kind, built_part, expected_part in (('coordinate', built.coords, expected.coords),
                                            ('variable', built.data_vars, expected.data_vars)):
        if set(built_part) != set(expected_part):
            found.append(f'{kind} names {sorted(built_part)} != {sorted(expected_part)}')
        for name in sorted(set(built_part) & set(expected_part)):
            if built_part[name].dims != expected_part[name].dims:
                found.append(f'{kind} {name}: dimensions {built_part[name].dims} != {expected_part[name].dims}')
            else:
                found += _array_differences(built_part[name].values, expected_part[name].values,
                                            f'{kind} {name}', exact=kind == 'coordinate')
    return found


def _json_differences(built, expected, where: str = '$') -> list:
    if type(built) is not type(expected):
        return [f'{where}: {type(built).__name__} != {type(expected).__name__}']
    if isinstance(expected, dict):
        if set(built) != set(expected):
            return [f'{where}: keys {sorted(built)} != {sorted(expected)}']
        return [found for key in expected for found in _json_differences(built[key], expected[key], f'{where}[{key!r}]')]
    if isinstance(expected, list):
        if len(built) != len(expected):
            return [f'{where}: {len(built)} items != {len(expected)}']
        return [found for index, (b, e) in enumerate(zip(built, expected))
                for found in _json_differences(b, e, f'{where}[{index}]')]
    if isinstance(expected, float):
        return _array_differences([built], [expected], where)
    return [] if built == expected else [f'{where}: {built!r} != {expected!r}']


def _differences(built, expected) -> list:
    """What differs between two parsed contents of the same kind (one comparison for each kind)."""
    if hasattr(expected, 'columns'):
        return _table_differences(built, expected)
    if hasattr(expected, 'data_vars'):
        return _netcdf_differences(built, expected)
    return _json_differences(built, expected)


def _content_digest(content) -> str:
    """The exact digest of a parsed content: the record that decides when no registered copy is here.

    JSON: its canonical text (types, structure and escaping are kept; a float's text round-trips
    to its exact bits). A table or netCDF file: for each column, coordinate and variable its name,
    dimensions, type and shape, then its values (numbers by their bytes, other values as JSON text).
    """
    if not hasattr(content, 'columns') and not hasattr(content, 'data_vars'):
        return hashlib.sha1(json.dumps(content, sort_keys=True).encode()).hexdigest()
    if hasattr(content, 'columns'):
        parts = [(f'column {column}', ['row'], content[column].to_numpy()) for column in content.columns]
    else:
        parts = [(f'{kind} {name}', list(map(str, part.dims)), part.values)
                 for kind, items in (('coordinate', content.coords), ('variable', content.data_vars))
                 for name, part in sorted(items.items(), key=lambda item: str(item[0]))]   # a map: no order
    digest = hashlib.sha1()
    for name, dims, values in parts:      # each header fixes the length of the values after it
        digest.update(json.dumps([name, dims, values.dtype.str, values.shape]).encode())
        digest.update(values.tobytes() if values.dtype.kind in 'biufc'
                      else json.dumps(values.ravel().tolist(), default=repr).encode())
    return digest.hexdigest()


def brainio_home() -> Path:
    """Where brainio keeps hosted files: ``$BRAINIO_HOME/<sha1>/<file name>``."""
    return Path(os.environ.get('BRAINIO_HOME') or '~/.brainio').expanduser().absolute()


def _packaging_folder(builder: DataBuilder) -> Path:
    """The plugin's ``data_packaging`` folder: the folder of the build script."""
    import importlib.util
    return Path(importlib.util.find_spec(shlex.split(builder.build)[2]).origin).parent


def _expected(name: str, builder: DataBuilder) -> dict:
    """``{file path: entry of build.json}`` that a cloud run recorded in ``data_packaging/expected.json``."""
    try:
        path = _packaging_folder(builder) / 'expected.json'
    except ModuleNotFoundError:          # a declaration whose script is not written yet records nothing
        return {}
    return json.loads(path.read_text()).get(name, {}) if path.exists() else {}


def _registered_copy(builder: DataBuilder, file: str, sha1: str) -> Path | None:
    """The registered file on this machine, found by its sha1: the brainio cache or the plugin folder."""
    for candidate in (brainio_home() / sha1 / file, _packaging_folder(builder).parent / file):
        if candidate.is_file() and _sha1(candidate) == sha1:
            return candidate
    return None


def _check(builder: DataBuilder, recorded: dict, relative: str, entry: dict, content) -> tuple:
    """``(outcome, problem)`` of one written file against its expected record; ``problem`` is '' or why it fails."""
    file = Path(relative).name
    sha1 = builder.sha1.get(file) or recorded.get('sha1')
    if not sha1:
        return f"no expected record: {entry['counts'] or str(entry['bytes']) + ' bytes'}", ''
    if sha1 == entry['sha1']:
        return 'identical', ''
    copy = _registered_copy(builder, file, sha1)
    if copy and content is not NO_PARSER:
        differences = _differences(content, _read(copy))
        if differences:
            return 'failed', f'its content differs from the registered copy {copy}: ' + '; '.join(
                differences[:5]) + (f'; and {len(differences) - 5} more' if len(differences) > 5 else '')
        return 'same content, bytes differ', ''
    if copy:
        return 'failed', f'its bytes differ from the registered copy {copy}, and only bytes are compared for this kind'
    if recorded.get('digest') and entry['digest']:
        if recorded['digest'] == entry['digest']:
            return 'same content, bytes differ', ''
        return 'failed', ('its content differs from the record, and no registered copy is on this machine to '
                          f'measure the difference: put the registered file at {brainio_home() / sha1 / file}')
    return 'failed', (f'its sha1 differs from the expected {sha1}, and no registered copy is on this machine '
                      f'to compare the content: put the registered file at {brainio_home() / sha1 / file} '
                      '(`brainscore.load_dataset` fetches a hosted file there), then run again with --output')


def package_data(name: str, source: str | Path | None = None,
                 output: str | Path | None = None) -> Path | None:
    """Package the files of data plugin ``name`` from its original source.

    Returns the output folder, or ``None`` when there is nothing to package.
    Raises ``LocalDataMissing`` with what to get and where to put it when the source
    is missing, and ``NotImplementedError`` when no builder exists yet.
    """
    data_home()             # a data home that names no absolute folder is refused before any folder is made
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
    if builder.reads == 'brainio' and destination == read_place:
        for file, sha1 in builder.sha1.items():     # the loader would refuse an entry that is not the file
            place = brainio_home() / sha1 / file
            if os.path.lexists(place) and not (place.is_file() and _sha1(place) == sha1):
                raise FileExistsError(f'{place} is not the registered file (sha1 {sha1}), and the loader refuses '
                                      'it. It is kept: move it away, then run again.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Build beside the destination and rename only after success: an interrupted
    # build must not look like a ready dataset.
    staging = Path(tempfile.mkdtemp(prefix='.brainscore-build-', dir=destination.parent))   # 0700, a new name
    work = Path(tempfile.mkdtemp(prefix='.brainscore-work-', dir=destination.parent))
    placed, keep = {}, False
    try:
        try:
            _run(builder.build, native, staging, work)
        except FileNotFoundError as missing:
            if builder.fetches:
                raise
            # A required input the script did not find: the source is incomplete.
            raise LocalDataMissing(_missing(name, builder, native, str(missing))) from missing
        staging.chmod(0o700)    # private again, whatever mode the script gave it
        keep = True             # from here a completed build is kept when the check fails or stops
        kept = (f'The built files are kept in this folder, which this call made; nothing reached {destination}:'
                f'\n    {staging}\nA new run of the command builds again. The folder can be deleted when it is '
                'no longer needed.')
        try:
            files, problems, recorded = {}, [], _expected(name, builder)
            for path in sorted(p for p in staging.rglob('*') if p.is_file()):
                if path.is_symlink():
                    raise ValueError('a data script must not create symlinks')
                relative = path.relative_to(staging).as_posix()
                content = _read(path)
                entry = files[relative] = {'bytes': path.stat().st_size, 'sha1': _sha1(path),
                                           'counts': '' if content is NO_PARSER else _counts(content),
                                           'digest': None if content is NO_PARSER else _content_digest(content)}
                entry['check'], problem = _check(builder, recorded.get(relative, {}), relative, entry, content)
                if problem:
                    problems.append(f'  {relative}: {problem}')
        except Exception as failure:
            raise RuntimeError(f'The check of the built {name} files stopped: {failure}\n{kept}') from failure
        if problems:
            raise ValueError('\n'.join([f'The check of the built {name} files failed.', *problems, kept]))
        (staging / 'build.json').write_text(json.dumps(
            {'schema_version': 4, 'name': name, 'source': builder.source, 'files': files},
            indent=2, allow_nan=False) + '\n')
        # Keep packaged data private by default: some sources forbid redistribution.
        for entry in [staging, *staging.rglob('*')]:
            entry.chmod(0o700 if entry.is_dir() else 0o600)
        keep = False
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('the output appeared during the build; nothing replaced')
        # Publish last, after the record and the permissions, so that little can fail after it. A hosted
        # file is published only when it is the registered file, so it stays when a later step fails.
        if builder.reads == 'brainio' and destination == read_place:
            _place_hosted(builder, staging, files, placed)
        os.rename(staging, destination)
    except BaseException as failure:
        if not placed:
            raise
        retained = ('These hosted files were published by this call as the registered files, and they stay:\n'
                    + '\n'.join(f'    {place}' for place in placed.values()))
        if not isinstance(failure, Exception):      # an interrupt stays an interrupt, with the list as a note
            failure.add_note(retained)
            raise
        raise RuntimeError(f'{failure}\n{retained}') from failure
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if staging.exists() and not keep:
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


def _place_hosted(builder: DataBuilder, staging: Path, files: dict, placed: dict) -> None:
    """Publish each identical hosted file at ``$BRAINIO_HOME/<sha1>/<file name>``; add it to ``placed``.

    ``os.link`` never replaces an entry (a broken link counts as one): the registered file already
    there is kept, any other entry stops the call. The staging folder is under ``$BRAINIO_HOME``.
    """
    for relative, entry in files.items():
        file = Path(relative).name
        if entry['check'] != 'identical' or file not in builder.sha1:
            continue
        place = brainio_home() / entry['sha1'] / file
        place.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.link(staging / relative, place)
        except FileExistsError:
            if place.is_file() and _sha1(place) == entry['sha1']:
                continue
            raise FileExistsError(f'{place} appeared during the build and is not the registered file; '
                                  'it is kept: move it away, then run again.') from None
        placed[relative] = place                  # recorded at once: the link is done
        os.unlink(staging / relative)


def _report(name: str, builder: DataBuilder, destination: Path, read_place: Path,
            files: dict, placed: dict) -> None:
    print(f'Packaged {name}.')
    for index, (relative, entry) in enumerate(files.items()):
        if index == _LISTED_FILES:
            print(f'  ... and {len(files) - index} more files (all in build.json)')
            break
        counts = entry['counts']
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
            identical = [relative for relative, entry in files.items() if entry['check'] == 'identical']
            other = [relative for relative in files if relative not in identical]
            step = f'The loader reads a hosted file only at {brainio_home()}/<sha1>/<file name>.'
            if identical:
                step += f' One step remains for {", ".join(identical)}: run again without --output to place it there.'
            if other:
                step += (f' The loader accepts only the registered bytes, so no packaging run makes {", ".join(other)} '
                         'readable; the benchmarks read the hosted file.')
        elif builder.reads:
            step = (f'The benchmarks read {read_place}; one step remains: move the folder there, '
                    'or put a link there to it.')
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
