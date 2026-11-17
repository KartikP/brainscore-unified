"""``package_data(name)``: every data plugin says how its files are packaged from the source.

The first two tests are an intended gate: a new data plugin fails them until it
registers a ``DataBuilder`` and its README carries the ``## Rebuild`` block of that
declaration, and every builder has a tiny made-up source below (also a gate). The
others run each builder through the public command on that source, with sockets,
network programs (git, aws) and S3 replaced by stand-ins whose uploads fail.
"""
import dataclasses
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shlex
import shutil
import socket
import stat
import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import brainscore
import brainscore.data
from brainscore import data_builder_registry
from brainscore.data import DataBuilder, package_data
from brainscore.data import local
from brainscore.data.local import LocalDataMissing

pytestmark = pytest.mark.unit

DATA = pathlib.Path(brainscore.data.__file__).parent
PLUGINS = sorted(p for p in DATA.iterdir() if (p / '__init__.py').exists() and not p.name.startswith('_'))


def _rebuild_blocks(readme: pathlib.Path) -> dict:
    """``{name: block}`` of the ``### <name>`` blocks of a README's ``## Rebuild`` section."""
    text = readme.read_text() if readme.exists() else ''
    if '## Rebuild' not in text:
        return {}
    section = re.split(r'^## (?!Rebuild)', text.split('## Rebuild', 1)[1], flags=re.M)[0]
    return {block.split('\n', 1)[0].strip(): ('### ' + block).strip()
            for block in section.split('### ')[1:]}


@pytest.mark.parametrize('plugin', PLUGINS, ids=lambda p: p.name)
def test_every_data_plugin_states_how_its_data_is_packaged(plugin):
    # Intended gate: a data plugin without these fails, and the message says what to add.
    blocks = _rebuild_blocks(plugin / 'README.md')
    assert blocks, (
        f'{plugin.name} does not say how its data is packaged. Register a factory in '
        f'brainscore/data/{plugin.name}/__init__.py: data_builder_registry[<name>] = lambda: DataBuilder(...) '
        f'(the builder command, or cannot=..., or nothing to package). Then add a section '
        f"'## Rebuild' to its README.md with the text of data_builder_registry[<name>]().readme(<name>). "
        f'See docs/guides/data.md.')
    for name, block in blocks.items():
        assert name in data_builder_registry, f'{plugin.name}/README.md names {name!r}, which no plugin registers'
        expected = data_builder_registry[name]().readme(name)
        assert block == expected, (
            f'{plugin.name}/README.md: the Rebuild block of {name} differs from its declaration. '
            f'Paste this block:\n\n{expected}')


def test_every_declaration_is_in_a_readme_and_answers_package_data():
    documented = set().union(*(_rebuild_blocks(plugin / 'README.md') for plugin in PLUGINS))
    for name, factory in data_builder_registry.items():
        builder = factory()
        assert name in documented, f'{name} has no Rebuild block in its plugin README.md'
        assert isinstance(builder, DataBuilder)
        assert not (builder.build and builder.cannot), f'{name} has a builder and says it cannot'
        if builder.build:
            module = builder.build.split('python -m ', 1)[1].split()[0]
            assert importlib.util.find_spec(module) is not None, f'{name}: {module} does not exist'


class _S3:
    """A stand-in S3 client: lists and downloads the objects a test gives; any upload fails."""

    def __init__(self, objects):
        self.objects = objects

    def get_paginator(self, operation):
        objects = self.objects
        return types.SimpleNamespace(paginate=lambda Bucket, Prefix: [
            {'Contents': [{'Key': key} for key in sorted(objects) if key.startswith(Prefix)]}])

    def download_file(self, bucket, key, path):
        pathlib.Path(path).write_bytes(self.objects[key])

    def __getattr__(self, name):           # upload_file, put_object, head_object, ...
        raise AssertionError(f'package_data called S3 {name}: nothing may be uploaded')


_NETWORK_PROGRAMS = {'git', 'aws', 'curl', 'wget', 'pip', 'hf', 'huggingface-cli', 'datalad', 'ssh',
                     'scp', 'rsync', 'gsutil'}


class _Offline:
    """Stand-ins a test fills: S3 objects, and programs other than ffmpeg (git, aws)."""

    def __init__(self):
        self.objects, self.programs = {}, {}


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No network: sockets fail, S3 is a stand-in whose uploads fail, and a program that
    reaches the network (git, aws, ...) runs only as a stand-in the test gives. The default source and output root
    is a temporary folder."""
    import subprocess
    stand_ins = _Offline()

    def refuse(*args, **kwargs):
        raise AssertionError('package_data used the network')
    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'create_connection', refuse)
    real_run = subprocess.run

    def run(argv, *args, **kwargs):
        program = pathlib.Path(str(argv[0] if isinstance(argv, (list, tuple)) else argv)).name
        if program in stand_ins.programs:
            return stand_ins.programs[program](list(map(str, argv)))
        if program in _NETWORK_PROGRAMS:
            raise AssertionError(f'package_data ran {argv}')
        return real_run(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'run', run)
    client = lambda *args, **kwargs: _S3(stand_ins.objects)
    boto3 = types.ModuleType('boto3')
    boto3.client = client
    boto3.session = types.SimpleNamespace(Config=lambda **kwargs: None, UNSIGNED=None)
    botocore = types.ModuleType('botocore')
    botocore.UNSIGNED = None
    botocore_config = types.ModuleType('botocore.config')
    botocore_config.Config = lambda **kwargs: None
    # The GIfTI reading of the Lahner time-series builder: nibabel.load(path).agg_data(), one array per TR.
    nibabel = types.ModuleType('nibabel')
    nibabel.load = lambda path: types.SimpleNamespace(agg_data=lambda: tuple(np.load(path)))
    for name, module in (('boto3', boto3), ('botocore', botocore), ('botocore.config', botocore_config),
                         ('nibabel', nibabel)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('BRAINIO_HOME', str(tmp_path / 'brainio'))
    return stand_ins


def _tree(folder):
    """Every file under ``folder`` with its sha1: a source must not change."""
    return {str(p.relative_to(folder)): _sha1(p) for p in sorted(folder.rglob('*')) if p.is_file()}


@pytest.mark.parametrize('folder', ['absent', 'empty'])
def test_a_missing_source_says_what_to_get_and_where_to_put_it(tmp_path, offline, folder):
    needs_source = [name for name, factory in data_builder_registry.items()
                    if factory().build and not factory().fetches]
    assert needs_source
    for name in needs_source:
        builder = data_builder_registry[name]()
        source = tmp_path / 'home' / 'sources' / name      # the one rule, the same for every plugin
        if folder == 'empty':
            source.mkdir(parents=True, exist_ok=True)
        with pytest.raises(LocalDataMissing) as missing:
            package_data(name, output=tmp_path / 'out' / name)
        message = str(missing.value)
        for fact in (builder.source, builder.access, builder.layout, builder.needs, str(source)):
            assert fact in message, f'{name}: the error lacks {fact!r}'
        assert not (tmp_path / 'out' / name).exists()


def test_a_plugin_with_no_builder_says_so(tmp_path):
    for name, factory in data_builder_registry.items():
        builder = factory()
        if builder.cannot:
            with pytest.raises(NotImplementedError) as refused:
                package_data(name, output=tmp_path / name)
            assert builder.cannot in str(refused.value)
        elif not builder.build:
            assert package_data(name, output=tmp_path / name) is None
        assert not (tmp_path / name).exists()


def _sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _without_record(name, monkeypatch):
    """Made-up data is not the real data: drop the expected record (the declaration's sha1 and expected.json)."""
    from brainscore.data import packaging
    expected = packaging._expected
    monkeypatch.setattr(packaging, '_expected', lambda record, builder: {} if record == name else expected(record, builder))
    builder = data_builder_registry[name]()
    if builder.sha1:
        monkeypatch.delitem(data_builder_registry, name)
        monkeypatch.setitem(data_builder_registry, name, lambda: dataclasses.replace(builder, sha1={}))


def test_ewok_packages_from_the_source_folder_of_the_one_rule(tmp_path, monkeypatch, capsys, offline):
    # Synthetic text only; no EWoK sentence is committed to the tests.
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path))
    source = tmp_path / 'sources' / 'ewok'
    source.mkdir(parents=True)
    pd.DataFrame([dict(Domain='synthetic', MetaTemplateID='1', TemplateID=str(i), Version='0',
                       Context1=f'context A {i}', Context2=f'context B {i}',
                       Target1=f'target A {i}', Target2=f'target B {i}') for i in range(3)]
                 ).to_csv(source / 'native.csv', index=False)
    output = package_data('ewok')
    # Where the benchmarks read it (here the folder its variable names), with nothing else to do.
    assert output == local.path('ewok-core-1.0') == tmp_path / 'ewok-core-1.0'
    assert len(brainscore.load_dataset('EWoK-core-1.0')) == 3
    printed = capsys.readouterr().out
    assert re.search(rf"items\.json .*sha1 {_sha1(output / 'items.json')} .*items 3", printed)
    assert _sha1(output / 'manifest.json') in printed
    assert printed.splitlines()[-3:] == [
        f'The data is now in: {output}', 'The benchmarks read it there. No more steps.',
        'Check: 2 no expected record.']


def _made_up_rajalingham():
    """A tiny 2-AFC data set: 6 objects, 11 images each, 5 distractors, 4 trials per condition."""
    objects = list('abcdef')
    rows = []
    for o, sample in enumerate(objects):
        for i in range(11):
            for distractor in objects:
                if distractor != sample:
                    rows += [(f'{sample}{i}', sample, distractor,
                              sample if (t + i + o) % (2 + i % 3) else distractor) for t in range(4)]
    image, sample, distractor, choice = map(list, zip(*rows))
    assembly = xr.DataArray(np.array(choice, dtype=object), dims=['presentation'], coords={
        'image_id': ('presentation', image), 'sample_obj': ('presentation', sample),
        'dist_obj': ('presentation', distractor)})
    stimuli = types.SimpleNamespace(stimulus_paths={i: f'{i}.png' for i in set(image)})
    return types.SimpleNamespace(load_dataset=lambda identifier: assembly,
                                 load_stimulus_set=lambda identifier: stimuli)


def test_rajalingham_checks_a_rebuild_against_the_files_its_tool_reads(tmp_path, monkeypatch, capsys, offline):
    original = data_builder_registry['rajalingham2018-2afc']()
    monkeypatch.setitem(sys.modules, 'brainscore_vision', _made_up_rajalingham())
    # The made-up rebuild differs from the registered files, which ship in the plugin folder: the check fails.
    with pytest.raises(ValueError, match=r'trials\.csv: its content differs from the registered copy .*rows'):
        package_data('rajalingham2018-2afc', output=tmp_path / 'first')
    assert not (tmp_path / 'first').exists()
    monkeypatch.setitem(data_builder_registry, 'rajalingham-fixture',
                        lambda: dataclasses.replace(original, sha1={}))
    first = package_data('rajalingham-fixture', output=tmp_path / 'first')
    printed = capsys.readouterr().out
    for file in ('trials.csv', 'tokens.csv'):
        rows = len(pd.read_csv(first / file))
        assert re.search(rf'{file} .*sha1 {_sha1(first / file)} .*rows {rows}\)', printed)
        assert f'check: no expected record: rows {rows}' in printed


def _made_up_plugin(tmp_path, monkeypatch, files, sha1=None, reads=''):
    """A plugin whose script copies its source folder: the test chooses the built files.

    Its ``data_packaging`` folder is ``tmp_path / 'scripts'``; its plugin folder is ``tmp_path``.
    """
    scripts = tmp_path / 'scripts'
    scripts.mkdir(exist_ok=True)
    (scripts / 'made_up_copy.py').write_text(
        'import shutil, sys\nshutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True)\n')
    monkeypatch.syspath_prepend(str(scripts))
    source = tmp_path / 'source'
    shutil.rmtree(source, ignore_errors=True)
    source.mkdir()
    for file, content in files.items():
        (source / file).write_bytes(content) if isinstance(content, bytes) else content.to_netcdf(source / file)
    if 'made-up' in data_builder_registry:
        return source
    monkeypatch.setitem(data_builder_registry, 'made-up', lambda: DataBuilder(
        source='made up', access='public', layout='any files', needs='nothing', writes='the same files',
        build='python -m made_up_copy {source} {output}', sha1=sha1 or {}, reads=reads))
    return source


def test_a_file_equal_to_its_registered_sha1_is_identical(tmp_path, monkeypatch, capsys, offline):
    table = b'x,y\n1.0,a\n2.0,b\n'
    source = _made_up_plugin(tmp_path, monkeypatch, {'table.csv': table},
                             sha1={'table.csv': hashlib.sha1(table).hexdigest()})
    package_data('made-up', source=source, output=tmp_path / 'out')
    assert re.search(r'table\.csv .*\n    check: identical\n', capsys.readouterr().out)


def _array(values, coordinate=None, dtype=float):
    """A one-variable netCDF assembly."""
    coordinate = list(range(len(values))) if coordinate is None else coordinate
    return xr.DataArray(np.array(values, dtype=dtype), dims=['presentation'],
                        coords={'stimulus_id': ('presentation', coordinate)}, name='data').to_dataset()


def _table(**columns):
    return pd.DataFrame(columns).to_csv(index=False).encode()


# (file, registered content, built content, the check passes): every pair has other bytes.
CONTENT_PAIRS = {
    'netcdf order': ('a.nc', _array([1, 2, 4, 3]), _array([1, 3, 2, 4]), False),
    'netcdf values': ('a.nc', _array([1, 4, 4]), _array([2, 2, 5]), False),
    'netcdf missing location': ('a.nc', _array([np.nan, 0]), _array([0, np.nan]), False),
    'netcdf one value outside its tolerance': ('a.nc', _array([1e6, 0]), _array([1e6, 0.4]), False),
    'netcdf coordinate type': ('a.nc', _array([1, 2], [1, 2]), _array([1, 2], ['1', '2']), False),
    'json float location': ('a.json', {'a': 1.0}, {'b': 1.0}, False),
    'json container type': ('a.json', {'a': []}, {'a': {}}, False),
    'json path': ('a.json', {'a/b': 1}, {'a': {'b': 1}}, False),
    'csv NA text': ('a.csv', b'x,y\nNA,1\nfoo,2\n', b'x,y\nnull,1\nfoo,2\n', False),
    'csv separator': ('a.csv', _table(x=['a\x1fb', 'c']), _table(x=['a', 'b\x1fc']), False),
    'csv integers as floats': ('a.csv', b'x\n1\n2\n', b'x\n1.0\n2.0\n', False),
    'csv equal content': ('a.csv', b'x,y\n1.0,a\n2.0,b\n', b'x,y\n1.00,a\n2.0000000001,b\n', True),
    'netcdf equal content': ('a.nc', _array([1.0, 2.0], [0.0, 1.0]), _array([1.0, 2.0000000001], [-0.0, 1.0]), True),
    'json equal content': ('a.json', {'a': [1.0, 'x']}, {'a': [1.0000000001, 'x']}, True),
    'netcdf infinity against a finite value': ('a.nc', _array([np.inf]), _array([0.0]), False),
    'netcdf opposite infinities': ('a.nc', _array([np.inf]), _array([-np.inf]), False),
    'json null equal content': ('a.json', b'null', b' null\n', True),
}


def _bytes(content, folder):
    """The bytes of a made-up file: given, a JSON structure, or a netCDF data set."""
    if isinstance(content, bytes):
        return content
    if isinstance(content, dict):
        return json.dumps(content).encode()
    content.to_netcdf(folder / 'written.nc')
    return (folder / 'written.nc').read_bytes()


@pytest.mark.parametrize('pair', sorted(CONTENT_PAIRS))
def test_other_bytes_pass_only_with_the_same_content_as_the_registered_copy(
        pair, tmp_path, monkeypatch, capsys, offline):
    file, registered, built, passes = CONTENT_PAIRS[pair]
    registered, built = _bytes(registered, tmp_path), _bytes(built, tmp_path)
    assert registered != built
    sha1 = hashlib.sha1(registered).hexdigest()
    (tmp_path / 'brainio' / sha1).mkdir(parents=True)          # the registered copy in the brainio cache
    (tmp_path / 'brainio' / sha1 / file).write_bytes(registered)
    source = _made_up_plugin(tmp_path, monkeypatch, {file: built}, sha1={file: sha1})
    if passes:
        package_data('made-up', source=source, output=tmp_path / 'out')
        assert 'check: same content, bytes differ' in capsys.readouterr().out
    else:
        with pytest.raises(ValueError, match=f'{file}: its content differs from the registered copy'):
            package_data('made-up', source=source, output=tmp_path / 'out')
        assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('passes', [True, False])
def test_with_no_copy_a_recorded_digest_decides(passes, tmp_path, monkeypatch, capsys, offline):
    source = _made_up_plugin(tmp_path, monkeypatch, {'a.nc': _bytes(_array([0.1, 0.2]), tmp_path)})
    first = package_data('made-up', source=source, output=tmp_path / 'first')
    # The record a cloud run copies from build.json into data_packaging/expected.json; no copy is here.
    record = json.loads((first / 'build.json').read_text())['files']
    record['a.nc']['sha1'] = '0' * 40
    (tmp_path / 'scripts' / 'expected.json').write_text(json.dumps({'made-up': record}))
    _made_up_plugin(tmp_path, monkeypatch, {'a.nc': _bytes(_array([0.1, 0.2 if passes else 0.2000001]), tmp_path)})
    if passes:
        package_data('made-up', source=source, output=tmp_path / 'second')
        assert 'check: same content, bytes differ' in capsys.readouterr().out
    else:
        with pytest.raises(ValueError, match='its content differs from the record, and no registered copy is on '
                                             'this machine to measure the difference'):
            package_data('made-up', source=source, output=tmp_path / 'second')
        assert not (tmp_path / 'second').exists()


def test_the_recorded_digest_does_not_depend_on_the_order_of_the_variables(tmp_path, monkeypatch, capsys, offline):
    def assembly(*names):
        return xr.Dataset({name: ('row', np.array(values)) for name, values in
                           (('a', [1.0, 2.0]), ('b', [3.0, 4.0])) if name in names}
                          )[list(names)]
    source = _made_up_plugin(tmp_path, monkeypatch, {'a.nc': _bytes(assembly('a', 'b'), tmp_path)})
    first = package_data('made-up', source=source, output=tmp_path / 'first')
    record = json.loads((first / 'build.json').read_text())['files']
    record['a.nc']['sha1'] = '0' * 40               # no registered copy: the digest decides
    (tmp_path / 'scripts' / 'expected.json').write_text(json.dumps({'made-up': record}))
    _made_up_plugin(tmp_path, monkeypatch, {'a.nc': _bytes(assembly('b', 'a'), tmp_path)})
    package_data('made-up', source=source, output=tmp_path / 'second')
    assert 'check: same content, bytes differ' in capsys.readouterr().out


def _hosted(tmp_path, monkeypatch, files):
    """A made-up plugin whose files are hosted: each registered sha1 is the sha1 of the built file."""
    sha1 = {file: hashlib.sha1(content).hexdigest() for file, content in files.items()}
    return _made_up_plugin(tmp_path, monkeypatch, files, sha1=sha1, reads='brainio'), sha1


@pytest.mark.parametrize('occupant', ['other bytes', 'folder', 'broken link'])
def test_an_occupied_cache_name_that_is_not_the_registered_file_is_refused_and_kept(
        occupant, tmp_path, monkeypatch, offline):
    source, sha1 = _hosted(tmp_path, monkeypatch, {'a.bin': b'registered'})
    place = tmp_path / 'brainio' / sha1['a.bin'] / 'a.bin'
    place.parent.mkdir(parents=True)
    {'other bytes': lambda: place.write_bytes(b'mine'), 'folder': place.mkdir,
     'broken link': lambda: place.symlink_to(tmp_path / 'absent')}[occupant]()
    with pytest.raises(FileExistsError, match=re.escape(str(place))):
        package_data('made-up', source=source)
    assert os.path.lexists(place) and (occupant != 'other bytes' or place.read_bytes() == b'mine')
    assert not (tmp_path / 'brainio' / 'packaged' / 'made-up').exists()


def _occupy_during_the_build(monkeypatch, places):
    """Put a file of the user at each place after the check, before the publication."""
    from brainscore.data import packaging
    check = packaging._check

    def occupying(*args, **kwargs):
        for place in places:
            place.parent.mkdir(parents=True, exist_ok=True)
            place.write_bytes(b'mine')
        return check(*args, **kwargs)
    monkeypatch.setattr(packaging, '_check', occupying)


def test_a_cache_name_occupied_during_the_build_is_not_replaced(tmp_path, monkeypatch, offline):
    source, sha1 = _hosted(tmp_path, monkeypatch, {'a.bin': b'registered'})
    place = tmp_path / 'brainio' / sha1['a.bin'] / 'a.bin'
    _occupy_during_the_build(monkeypatch, [place])
    with pytest.raises(FileExistsError, match=re.escape(str(place))):
        package_data('made-up', source=source)
    assert place.read_bytes() == b'mine'


@pytest.mark.parametrize('failure', ['none', 'second publication', 'record folder'])
def test_a_failure_after_a_publication_deletes_or_replaces_nothing_under_brainio_home(
        failure, tmp_path, monkeypatch, offline):
    source, sha1 = _hosted(tmp_path, monkeypatch, {'a.bin': b'registered a', 'b.bin': b'registered b'})
    places = {file: tmp_path / 'brainio' / value / file for file, value in sha1.items()}
    if failure == 'second publication':           # a user puts a file at the second name during the build
        _occupy_during_the_build(monkeypatch, [places['b.bin']])
    if failure == 'record folder':
        def refuse(*args):
            raise OSError('the record folder cannot be renamed')
        monkeypatch.setattr(os, 'rename', refuse)
    if failure == 'none':
        package_data('made-up', source=source)
        assert all(stat.S_IMODE(place.stat().st_mode) == 0o600 for place in places.values())   # private
        return
    with pytest.raises(RuntimeError, match='published by this call') as failed:
        package_data('made-up', source=source)
    assert str(places['a.bin']) in str(failed.value)
    assert places['a.bin'].read_bytes() == b'registered a'                 # the registered file stays
    if failure == 'second publication':
        assert places['b.bin'].read_bytes() == b'mine'                     # the user's file is untouched
    assert not (tmp_path / 'brainio' / 'packaged' / 'made-up').exists()


def test_a_staging_unlink_error_after_a_link_names_the_published_file(tmp_path, monkeypatch, offline):
    source, sha1 = _hosted(tmp_path, monkeypatch, {'a.bin': b'registered a'})
    place = tmp_path / 'brainio' / sha1['a.bin'] / 'a.bin'
    unlink = os.unlink

    def refuse(path, *args, **kwargs):
        if '.brainscore-build-' in str(path):
            raise OSError('staging unlink denied')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(os, 'unlink', refuse)
    with pytest.raises(RuntimeError, match='staging unlink denied') as failed:
        package_data('made-up', source=source)
    assert str(place) in str(failed.value) and place.read_bytes() == b'registered a'


def test_an_interrupt_after_a_publication_names_the_published_file(tmp_path, monkeypatch, offline):
    source, sha1 = _hosted(tmp_path, monkeypatch, {'a.bin': b'registered a'})
    place = tmp_path / 'brainio' / sha1['a.bin'] / 'a.bin'

    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(os, 'rename', interrupt)
    with pytest.raises(KeyboardInterrupt) as interrupted:
        package_data('made-up', source=source)
    assert str(place) in '\n'.join(getattr(interrupted.value, '__notes__', []))
    assert place.read_bytes() == b'registered a'


def test_a_relative_brainio_home_is_printed_as_an_absolute_place(tmp_path, monkeypatch, capsys, offline):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('BRAINIO_HOME', 'relative-cache')
    source, sha1 = _hosted(tmp_path, monkeypatch, {'a.bin': b'registered'})
    package_data('made-up', source=source)
    assert f"    {pathlib.Path.cwd() / 'relative-cache' / sha1['a.bin'] / 'a.bin'}\n" in capsys.readouterr().out


@pytest.mark.parametrize('same_bytes', [True, False])
def test_an_explicit_hosted_output_states_the_step_its_outcome_allows(same_bytes, tmp_path, monkeypatch, capsys, offline):
    registered, built = b'x\n1.0\n', b'x\n1.0\n' if same_bytes else b'x\n1.00\n'
    sha1 = hashlib.sha1(registered).hexdigest()
    (tmp_path / 'brainio' / sha1).mkdir(parents=True)
    (tmp_path / 'brainio' / sha1 / 'a.csv').write_bytes(registered)
    source = _made_up_plugin(tmp_path, monkeypatch, {'a.csv': built}, sha1={'a.csv': sha1}, reads='brainio')
    package_data('made-up', source=source, output=tmp_path / 'out')
    printed = capsys.readouterr().out
    assert ('run again without --output' in printed) == same_bytes
    assert ('no packaging run makes a.csv readable' in printed) != same_bytes


def test_the_ewok_error_gives_the_command_that_writes_where_it_reads(offline):
    from brainscore.data.ewok.data import load_prepared
    with pytest.raises(FileNotFoundError) as missing:
        load_prepared()
    assert str(local.REGISTRY['ewok-core-1.0'].resolved()) in str(missing.value)
    assert 'package ewok' in str(missing.value) and '--output' not in str(missing.value)


def test_a_failed_check_keeps_the_build_and_a_new_run_builds_again(tmp_path, monkeypatch, offline):
    registered = b'x,y\n1.0,a\n'
    sha1 = hashlib.sha1(registered).hexdigest()
    source = _made_up_plugin(tmp_path, monkeypatch, {'table.csv': b'x,y\n1.00,a\n'}, sha1={'table.csv': sha1})

    def kept_folder():
        with pytest.raises(ValueError) as failed:      # no registered copy here: the check fails
            package_data('made-up', source=source, output=tmp_path / 'out')
        lines = str(failed.value).splitlines()
        kept = pathlib.Path(lines[lines.index(next(line for line in lines if 'kept in this folder' in line)) + 1].strip())
        assert 'A new run of the command builds again' in str(failed.value)
        assert 'rm ' not in str(failed.value)                                  # no removal command
        return kept
    first = kept_folder()
    assert first.is_absolute() and stat.S_IMODE(first.stat().st_mode) == 0o700
    assert (first / 'table.csv').read_bytes() == b'x,y\n1.00,a\n'
    assert not (tmp_path / 'out').exists()                                    # the read place is untouched
    before = {path.name: path.read_bytes() for path in first.iterdir()}
    second = kept_folder()                                                    # a new run builds again
    assert second != first and {path.name: path.read_bytes() for path in first.iterdir()} == before


def test_a_failure_after_the_check_passed_keeps_no_build(tmp_path, monkeypatch, offline):
    source = _made_up_plugin(tmp_path, monkeypatch, {'table.csv': b'x\n1\n'})
    write_text = pathlib.Path.write_text

    def refuse(self, *args, **kwargs):
        if self.name == 'build.json':
            raise OSError('the record cannot be written')
        return write_text(self, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, 'write_text', refuse)
    with pytest.raises(OSError, match='the record cannot be written'):
        package_data('made-up', source=source, output=tmp_path / 'out')
    assert not list(tmp_path.glob('.brainscore-*')) and not (tmp_path / 'out').exists()


def test_a_file_with_no_expected_record_states_its_counts(tmp_path, monkeypatch, capsys, offline):
    source = _made_up_plugin(tmp_path, monkeypatch, {'table.csv': b'x\n1\n2\n'})
    output = package_data('made-up', source=source, output=tmp_path / 'out')
    assert 'check: no expected record: rows 2' in capsys.readouterr().out
    assert json.loads((output / 'build.json').read_text())['files']['table.csv']['check'] == \
        'no expected record: rows 2'


def _gifti(path, n_tr):
    """A made-up surface time series: ``n_tr`` arrays over a full fsaverage hemisphere, stored as the
    nibabel stand-in of ``offline`` reads it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        np.save(handle, np.stack([np.full(10242 + 5, t, dtype=np.float32) for t in range(n_tr)]))


def _lahner_source(source):
    """One run of ds005165 laid out as the bucket, with a file of the user beside it."""
    func = source / 'derivatives/versionB/fmriprep/sub-01/ses-02/func'
    stem = 'sub-01_ses-02_task-test_run-1'
    for hemi in 'LR':
        _gifti(func / f'{stem}_hemi-{hemi}_space-fsaverage_bold.func.gii', n_tr=4)
    events = source / 'sub-01/ses-02/func' / f'{stem}_events.tsv'
    events.parent.mkdir(parents=True)
    pd.DataFrame({'onset': [0.0, 4.0], 'duration': [3.0, 3.0], 'trial_type': ['test', 'oddball'],
                  'stim_file': ['test/0001.mp4', 'n/a']}).to_csv(events, sep='\t', index=False)
    (source / 'derivatives/versionB/fmriprep/sub-01/my-notes.txt').write_text('mine')
    (source / 'run_records.pkl').write_bytes(b'an older checkpoint of mine')
    return f'ds005165/derivatives/versionB/fmriprep/sub-01/ses-02/func/{stem}_hemi-L_space-fsaverage_bold.func.gii'


@pytest.mark.parametrize('listed', [True, False], ids=['builds', 'fails'])
def test_lahner_time_series_reads_the_source_and_never_changes_it(tmp_path, monkeypatch, offline, listed):
    _without_record('lahner2024-timeresolved', monkeypatch)
    source = tmp_path / 'ds005165'
    key = _lahner_source(source)
    before = _tree(source)
    # The bucket listing names the run (or nothing); a download of a file the source holds fails.
    def aws(argv):
        if argv[1:3] != ['s3', 'ls']:
            pytest.fail(f'downloaded a file the source holds: {argv}')
        named = listed and argv[-1].endswith('/sub-01/') and 'task-test' in key
        return types.SimpleNamespace(stdout=f'2026-01-01 00:00:00 1 {key}\n' if named else '', returncode=0)
    offline.programs['aws'] = aws
    built = [tmp_path / 'out']
    if listed:
        output = package_data('lahner2024-timeresolved', source=source, output=built[0])
        assert (output / 'assy_Lahner2024-fMRI-timeresolved.nc').exists()
    else:
        with pytest.raises(RuntimeError):
            package_data('lahner2024-timeresolved', source=source, output=built[0])
        assert not built[0].exists()
    assert _tree(source) == before
    assert not list(tmp_path.glob('.brainscore-*'))


def _algonauts_source(root, participants=(1, 2, 3, 5), film=b''):
    """A tiny competition tree: one Friends film of 3 scans, one S7 and one OOD film."""
    import h5py
    for participant in participants:
        sub = f'sub-{participant:02d}'
        (root / 'fmri' / sub / 'func').mkdir(parents=True)
        with h5py.File(root / 'fmri' / sub / 'func' / f'{sub}_task-friends_s1_bold.h5', 'w') as h5:
            h5['ses-001_task-s01e01a'] = np.full((3, 1000), participant, dtype=np.float32)
        counts = root / 'fmri' / sub / 'target_sample_number'
        counts.mkdir()
        np.save(counts / f'{sub}_friends-s7_fmri_samples.npy', {'s07e01a': 2})
        np.save(counts / f'{sub}_ood_fmri_samples.npy', {'chaplin1': 2})
    films = {'friends/s1/friends_s01e01a.mkv': film, 'friends/s7/friends_s07e01a.mkv': b'',
             'ood/chaplin/task-chaplin1_video.mkv': b''}
    for relative, content in films.items():
        (root / 'stimuli/movies' / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / 'stimuli/movies' / relative).write_bytes(content)
    (root / 'stimuli/movies/movie10').mkdir()


def test_algonauts_fails_as_a_whole_when_a_participant_lacks_its_input(tmp_path, offline):
    _algonauts_source(tmp_path / 'native', participants=(1, 3, 5))
    with pytest.raises(LocalDataMissing) as missing:
        package_data('algonauts2025', source=tmp_path / 'native', output=tmp_path / 'out')
    assert 'algonauts2025_friends_sub02.nc' in str(missing.value)
    assert data_builder_registry['algonauts2025']().access in str(missing.value)
    assert not (tmp_path / 'out').exists()


@pytest.mark.skipif(__import__('shutil').which('ffmpeg') is None, reason='ffmpeg is needed')
def test_algonauts_builds_every_file_the_registered_benchmarks_read(tmp_path, monkeypatch, offline):
    _without_record('algonauts2025', monkeypatch)
    import subprocess
    from brainscore.data.algonauts2025 import per_scan_stimuli, scans_per_stimulus
    film = tmp_path / 'film.mkv'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=duration=5:size=64x48:rate=25',
                    '-f', 'lavfi', '-i', 'sine=duration=5:sample_rate=16000', '-shortest',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(film)], check=True)
    _algonauts_source(tmp_path / 'native', film=film.read_bytes())
    output = package_data('algonauts2025', source=tmp_path / 'native', output=tmp_path / 'out')
    # Read the result as the benchmarks do: the registered loaders, then the media table of each variant.
    for participant, media in ((1, ['frames']), (1, ['clips']), (1, ['clips', 'sound']), (2, ['frames']),
                               (3, ['frames']), (5, ['frames'])):
        assembly = brainscore.load_dataset(f'Algonauts2025-friends-sub{participant:02d}', root=output)
        films = brainscore.load_stimulus_set('Algonauts2025-friends', root=output)
        table = per_scan_stimuli(films, scans_per_stimulus(assembly), media, 'friends', participant, output)
        assert len(table) == assembly.sizes['presentation'] == 3


@pytest.mark.skipif(__import__('shutil').which('ffmpeg') is None, reason='ffmpeg is needed')
def test_the_algonauts_recovery_command_reads_the_folder_the_benchmark_names(tmp_path, monkeypatch, offline):
    _without_record('algonauts2025', monkeypatch)
    import subprocess
    from brainscore.data.__main__ import main
    film = tmp_path / 'film.mkv'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=duration=5:size=64x48:rate=25',
                    '-f', 'lavfi', '-i', 'sine=duration=5:sample_rate=16000', '-shortest',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(film)], check=True)
    # The competition files are in the one source folder.
    _algonauts_source(tmp_path / 'home' / 'sources' / 'algonauts2025', film=film.read_bytes())
    prepared = local.REGISTRY['algonauts2025-prepared'].resolved()
    with pytest.raises(LocalDataMissing) as missing:
        local.check_benchmark_assets('Algonauts2025-friends-sub01')
    # Run the command the message gives, as written, with the folder it names.
    command = str(missing.value).split('Then convert it:')[1].split('\n')[1].strip()
    assert main(shlex.split(command)[3:]) == 0
    local.check_benchmark_assets('Algonauts2025-friends-sub01')
    assert brainscore.load_dataset('Algonauts2025-friends-sub01', root=prepared).sizes['presentation'] == 3


@pytest.mark.skipif(__import__('shutil').which('ffmpeg') is None, reason='ffmpeg is needed')
def test_algonauts_packaged_from_the_common_source_folder_passes_the_benchmark_check(
        tmp_path, monkeypatch, offline):
    _without_record('algonauts2025', monkeypatch)
    # Two steps: the source in the one source folder, package_data, then the benchmark's pre-check.
    source = _algonauts_film_source(tmp_path, monkeypatch, offline)
    (tmp_path / 'home' / 'sources').mkdir(parents=True)
    source.rename(tmp_path / 'home' / 'sources' / 'algonauts2025')
    package_data('algonauts2025')
    assert (tmp_path / 'home' / 'algonauts2025-prepared').is_dir()
    local.check_benchmark_assets('Algonauts2025-friends-sub01')


def _clips(folder, monkeypatch):
    """Two tiny hosted clips, one with sound and one without, as ``load_stimulus_set`` gives them."""
    import subprocess
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    folder.mkdir()
    paths = {'stimulus_0001': folder / '0001.mp4', 'stimulus_0002': folder / '0002.mp4'}
    for path, sound in zip(paths.values(), (True, False)):
        inputs = ['-f', 'lavfi', '-i', 'testsrc=duration=3:size=64x48:rate=25']
        if sound:
            inputs += ['-f', 'lavfi', '-i', 'sine=duration=3:sample_rate=16000']
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', *inputs, '-shortest', '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', str(path)], check=True)
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': list(paths)}))
    stimuli.identifier = 'BOLDMoments'
    stimuli.stimulus_paths = {key: str(path) for key, path in paths.items()}
    monkeypatch.setattr(brainscore, 'load_stimulus_set', lambda identifier: stimuli)
    return paths


@pytest.mark.skipif(__import__('shutil').which('ffmpeg') is None, reason='ffmpeg is needed')
def test_the_lahner_sound_manifest_names_files_that_exist_after_the_build(tmp_path, monkeypatch, offline):
    import json
    _clips(tmp_path / 'clips', monkeypatch)
    output = package_data('lahner2024-sound', output=tmp_path / 'sound')
    manifest = json.loads((output / 'manifest.json').read_text())
    assert len(manifest['entries']) == 2
    for entry in manifest['entries']:
        assert (output / entry['audio_path']).is_file()


# --- every builder through the public command --------------------------------------
# A tiny made-up source for each builder: (source folder or None when the script fetches it).

def _ewok_source(tmp_path, monkeypatch, offline):
    source = tmp_path / 'ewok'
    source.mkdir()
    pd.DataFrame([dict(Domain='synthetic', MetaTemplateID='1', TemplateID='0', Version='0',
                       Context1='context A', Context2='context B', Target1='target A', Target2='target B')]
                 ).to_csv(source / 'native.csv', index=False)
    return source


def _lebel_source(tmp_path, monkeypatch, offline):
    import pickle
    n_tr = 6
    story = types.SimpleNamespace(brain_data=np.arange(n_tr * 4, dtype=np.float32).reshape(n_tr, 4),
                                  tr_times=np.arange(n_tr + 15) * 2.0, words=['one', 'two', 'three'],
                                  data_times=[21.0, 23.0, 25.0])
    source = tmp_path / 'lebel'
    source.mkdir()
    (source / 'assembly_lebel_uts03.pkl').write_bytes(
        pickle.dumps(types.SimpleNamespace(stories=['story'], story_data={'story': story})))
    return source


def _algonauts_film_source(tmp_path, monkeypatch, offline):
    import subprocess
    film = tmp_path / 'film.mkv'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=duration=5:size=64x48:rate=25',
                    '-f', 'lavfi', '-i', 'sine=duration=5:sample_rate=16000', '-shortest',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(film)], check=True)
    _algonauts_source(tmp_path / 'native', film=film.read_bytes())
    return tmp_path / 'native'


def _lahner_clips_source(tmp_path, monkeypatch, offline):
    _clips(tmp_path / 'clips', monkeypatch)


def _lahner_series_source(tmp_path, monkeypatch, offline):
    key = _lahner_source(tmp_path / 'ds005165')

    def aws(argv):
        assert argv[1:3] == ['s3', 'ls'], argv
        named = argv[-1].endswith('/sub-01/')
        return types.SimpleNamespace(stdout=f'2026-01-01 00:00:00 1 {key}\n' if named else '', returncode=0)
    offline.programs['aws'] = aws
    return tmp_path / 'ds005165'


def _lahner_motion_source(tmp_path, monkeypatch, offline):
    tsv = pd.DataFrame({column: [0.0, 0.1] for column in (
        'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter')})
    tsv['framewise_displacement'] = ['n/a', 0.2]
    offline.objects['ds005165/derivatives/versionB/fmriprep/sub-01/ses-02/func/'
                    'sub-01_ses-02_task-test_run-1_desc-confounds_timeseries.tsv'] = \
        tsv.to_csv(sep='\t', index=False).encode()


def _rajalingham_source(tmp_path, monkeypatch, offline):
    monkeypatch.setitem(sys.modules, 'brainscore_vision', _made_up_rajalingham())


SOURCES = {'ewok': _ewok_source, 'lebel2023': _lebel_source, 'algonauts2025': _algonauts_film_source,
           'lahner2024-frames': _lahner_clips_source,
           'lahner2024-sound': _lahner_clips_source, 'lahner2024-timeresolved': _lahner_series_source,
           'lahner2024-motion': _lahner_motion_source, 'rajalingham2018-2afc': _rajalingham_source}
UPLOAD_FLAGS = {'lebel2023': '--upload', 'lahner2024-timeresolved': '--upload-to-s3',
                'lahner2024-motion': '--upload-to-s3'}


def test_every_builder_has_a_made_up_source_here():
    # Intended gate: a new builder gets a tiny made-up source in SOURCES, so the test below runs it.
    builders = {name for name, factory in data_builder_registry.items() if factory().build}
    assert builders <= set(SOURCES), f'add a tiny made-up source to SOURCES for {sorted(builders - set(SOURCES))}'


def _package(name, tmp_path, monkeypatch, offline):
    from brainscore.data.__main__ import main
    source = SOURCES[name](tmp_path, monkeypatch, offline)
    _without_record(name, monkeypatch)
    arguments = ['package', name, '--output', str(tmp_path / 'out')]
    return main(arguments + (['--source', str(source)] if source else []))


@pytest.mark.skipif(__import__('shutil').which('ffmpeg') is None, reason='ffmpeg is needed')
@pytest.mark.parametrize('name', sorted(SOURCES))
def test_every_builder_runs_through_the_public_command_and_uploads_nothing(name, tmp_path, monkeypatch, offline):
    assert _package(name, tmp_path, monkeypatch, offline) == 0
    written = [p for p in (tmp_path / 'out').rglob('*') if p.is_file() and p.name != 'build.json']
    assert written, f'{name} wrote nothing'


@pytest.mark.parametrize('name', sorted(UPLOAD_FLAGS))
def test_a_planted_upload_fails(name, tmp_path, monkeypatch, offline):
    builder = data_builder_registry[name]()
    monkeypatch.delitem(data_builder_registry, name)
    monkeypatch.setitem(data_builder_registry, name,
                        lambda: dataclasses.replace(builder, build=f'{builder.build} {UPLOAD_FLAGS[name]}'))
    with pytest.raises(AssertionError, match='nothing may be uploaded'):
        _package(name, tmp_path, monkeypatch, offline)
    assert not (tmp_path / 'out').exists()


def test_the_lebel_source_pickle_is_read_in_the_source_folder_of_the_data_home(tmp_path, monkeypatch, offline):
    from brainscore.data.lebel2023 import data
    with pytest.raises(LocalDataMissing, match=re.escape(str(tmp_path / 'home' / 'sources' / 'lebel2023'))):
        data.build()
    source = _lebel_source(tmp_path, monkeypatch, offline)
    (tmp_path / 'home' / 'sources').mkdir(parents=True)
    source.rename(tmp_path / 'home' / 'sources' / 'lebel2023')     # the folder package_data reads
    stimuli, assembly = data.build()
    assert len(stimuli) == assembly.sizes['presentation'] == 6


def test_a_rebuilt_hosted_file_goes_where_the_loader_reads_it(tmp_path, monkeypatch, capsys, offline):
    from brainscore.data.lebel2023 import hosted
    from brainscore_core.supported_data_standards.brainio import fetch
    from brainscore_core.supported_data_standards.brainio.s3 import get_path
    builder = data_builder_registry['lebel2023']()
    assert _package('lebel2023', tmp_path, monkeypatch, offline) == 0     # --output: learn the made-up sha1
    built = {file: _sha1(tmp_path / 'out' / file) for file in builder.sha1}
    monkeypatch.delitem(data_builder_registry, 'lebel2023')
    monkeypatch.setitem(data_builder_registry, 'lebel2023', lambda: dataclasses.replace(builder, sha1=built))
    capsys.readouterr()
    record = package_data('lebel2023', source=tmp_path / 'lebel')
    assert record == tmp_path / 'brainio' / 'packaged' / 'lebel2023'
    assert sorted(p.name for p in record.iterdir()) == ['build.json']
    printed = capsys.readouterr().out
    for file, sha1 in built.items():
        assert f"    {tmp_path / 'brainio' / sha1 / file}" in printed
    assert 'The loader reads each placed file there in place of the download. No more steps.' in printed
    # The loader reads it with no download.
    monkeypatch.setattr(fetch, 'get_local_data_path', lambda: str(tmp_path / 'brainio'))
    file = f'stimulus_{hosted.TR_STIMULI_ID}.csv'
    path = get_path(hosted.TR_STIMULI_ID, 'csv', hosted.BUCKET, hosted.VERSION_ID['TR stimuli'], built[file],
                    filename_prefix='stimulus_', folder_name=hosted.FOLDER)
    assert _sha1(pathlib.Path(path)) == built[file]


@pytest.mark.skipif(__import__('shutil').which('ffmpeg') is None, reason='ffmpeg is needed')
def test_algonauts_fails_as_a_whole_when_an_assembly_names_an_absent_film(tmp_path, offline):
    import h5py
    source = _algonauts_film_source(tmp_path, None, offline)
    # The fMRI of every participant also holds a second film, whose file the source lacks.
    for h5_path in source.glob('fmri/sub-0*/func/*.h5'):
        with h5py.File(h5_path, 'a') as h5:
            h5['ses-001_task-s01e02a'] = np.zeros((3, 1000), dtype=np.float32)
    before = _tree(source)
    with pytest.raises(LocalDataMissing) as missing:
        package_data('algonauts2025', source=source, output=tmp_path / 'out')
    assert 'friends_s01e02a' in str(missing.value)
    assert not (tmp_path / 'out').exists()
    assert not list(tmp_path.glob('.brainscore-*'))
    assert _tree(source) == before


def test_missing_prepared_data_names_its_place_in_the_data_home_and_the_command(tmp_path, offline):
    with pytest.raises(LocalDataMissing) as missing:
        local.check_benchmark_assets('Algonauts2025-friends-sub01')
    assert str(tmp_path / 'home' / 'algonauts2025-prepared') in str(missing.value)
    command = str(missing.value).split('Then convert it:')[1].split('\n')[1].strip()
    assert command == 'python -m brainscore.data package algonauts2025'


@pytest.mark.parametrize('record', sorted(DATA.glob('*/data_packaging/expected.json')),
                         ids=lambda path: path.parts[-3])
def test_every_expected_record_holds_what_the_check_reads(record):
    # The entries a cloud run copied from build.json: each is keyed by a registered name, then by a file.
    entries = json.loads(record.read_text())
    assert entries
    for name, files in entries.items():
        assert name in data_builder_registry, f'{record}: {name} is not a registered name'
        assert files, f'{record}: {name} records no file'
        for file, entry in files.items():
            assert re.fullmatch(r'[0-9a-f]{40}', entry['sha1']), f'{record}: {file} has no sha1'
            assert entry['digest'] is None or re.fullmatch(r'[0-9a-f]{40}', entry['digest']), \
                f'{record}: {file} has no exact digest'


def test_the_package_ships_each_data_plugin_readme_and_expected_record(tmp_path):
    # The Rebuild gate above reads the READMEs from the imported package, so the wheel must hold them.
    from tests.test_coggym_packaging import _packaged_files
    names = _packaged_files(tmp_path)
    wanted = [f'brainscore/data/{plugin.name}/README.md' for plugin in PLUGINS]
    wanted += [path.relative_to(DATA.parents[1]).as_posix() for path in DATA.glob('*/data_packaging/expected.json')]
    assert [name for name in wanted if name not in names] == []


def test_a_printed_command_quotes_every_path(tmp_path, monkeypatch, offline):
    # Folders with spaces: a command line that names one must quote it, or the message prints no command.
    shared = tmp_path / 'shared data'
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(shared / 'home'))
    registered = b'x\n1.0\n'
    source = _made_up_plugin(tmp_path, monkeypatch, {'table.csv': b'x\n1.00\n'},
                             sha1={'table.csv': hashlib.sha1(registered).hexdigest()})
    messages = []
    with pytest.raises(ValueError) as failed:                     # a failed check
        package_data('made-up', source=source, output=shared / 'out')
    messages.append(str(failed.value))
    with pytest.raises(LocalDataMissing) as missing:              # a missing source
        package_data('ewok', source=shared / 'no source')
    messages.append(str(missing.value))
    for asset in local.REGISTRY.values():                         # a missing local asset
        messages.append(asset.instructions())
    for message in messages:
        for line in message.splitlines():
            if re.search(r'\b(python -m|rm|mv|cp|mkdir)\s', line):
                for path in re.findall(rf"{re.escape(str(shared))}[^'\n]*", line):
                    assert shlex.quote(path.rstrip('.')) in line, f'unquoted path in: {line}'
