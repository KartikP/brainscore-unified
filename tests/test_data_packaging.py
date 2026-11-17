"""``package_data(name)``: every data plugin says how its files are packaged from the source.

The first two tests are an intended gate: a new data plugin fails them until it
registers a ``DataBuilder`` and its README carries the ``## Rebuild`` block of that
declaration. The others run two builders end to end on tiny made-up sources, with
the network replaced by a function that fails.
"""
import dataclasses
import hashlib
import importlib.util
import pathlib
import re
import socket
import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import brainscore.data
from brainscore import data_builder_registry
from brainscore.data import DataBuilder, package_data
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
        f'{plugin.name} does not say how its data is packaged. Register a DataBuilder in '
        f'brainscore/data/{plugin.name}/__init__.py (data_builder_registry[<name>] = DataBuilder(...): '
        f'the builder command, or cannot=..., or nothing to package), then add a section '
        f"'## Rebuild' to its README.md with DataBuilder.readme(<name>). See docs/guides/data.md.")
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


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('package_data used the network')
    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'create_connection', refuse)


def test_a_missing_source_says_what_to_get_and_where_to_put_it(tmp_path, monkeypatch, no_network):
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path / 'home'))
    needs_source = [name for name, factory in data_builder_registry.items()
                    if factory().build and not factory().fetches]
    assert needs_source
    for name in needs_source:
        builder = data_builder_registry[name]()
        with pytest.raises(LocalDataMissing) as missing:
            package_data(name)
        message = str(missing.value)
        for fact in (builder.source, builder.access, builder.layout, builder.needs,
                     str(tmp_path / 'home' / 'sources' / name)):
            assert fact in message, f'{name}: the error lacks {fact!r}'
        assert not (tmp_path / 'home' / 'packaged' / name).exists()


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


def test_ewok_packages_from_the_source_folder_of_the_one_rule(tmp_path, monkeypatch, capsys, no_network):
    # Synthetic text only; no EWoK sentence is committed to the tests.
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path))
    source = tmp_path / 'sources' / 'ewok'
    source.mkdir(parents=True)
    pd.DataFrame([dict(Domain='synthetic', MetaTemplateID='1', TemplateID=str(i), Version='0',
                       Context1=f'context A {i}', Context2=f'context B {i}',
                       Target1=f'target A {i}', Target2=f'target B {i}') for i in range(3)]
                 ).to_csv(source / 'native.csv', index=False)
    output = package_data('ewok')
    assert output == tmp_path / 'packaged' / 'ewok'
    printed = capsys.readouterr().out
    assert re.search(rf"items\.json .*sha1 {_sha1(output / 'items.json')} .*items 3", printed)
    assert _sha1(output / 'manifest.json') in printed


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


def test_rajalingham_prints_sha1_and_counts_and_names_the_cache_path_when_equal(
        tmp_path, monkeypatch, capsys, no_network):
    monkeypatch.setitem(sys.modules, 'brainscore_vision', _made_up_rajalingham())
    first = package_data('rajalingham2018-2afc', output=tmp_path / 'first')
    printed = capsys.readouterr().out
    for file in ('trials.csv', 'tokens.csv'):
        rows = len(pd.read_csv(first / file))
        assert re.search(rf'{file} .*sha1 {_sha1(first / file)} .*rows {rows}\)', printed)
    assert 'differs from the registered sha1' in printed          # made-up data is not the registered data
    # A rebuild that equals the registered sha1 is told where brainio uses it with no download.
    registered = _sha1(first / 'trials.csv')
    original = data_builder_registry['rajalingham2018-2afc']()
    monkeypatch.setitem(data_builder_registry, 'rajalingham-fixture',
                        lambda: dataclasses.replace(original, sha1={'trials.csv': registered}))
    package_data('rajalingham-fixture', output=tmp_path / 'second')
    assert f'$BRAINIO_HOME/{registered}/trials.csv' in capsys.readouterr().out
