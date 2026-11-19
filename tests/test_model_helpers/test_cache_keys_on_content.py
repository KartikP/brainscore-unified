"""Guard: a store that Brain-Score ships finds a stored result by the content of its input.

Two parts, both found by discovery, with no list of plugins or functions:

1. Every function in unified ``brainscore/`` and in the shared vision
   ``model_helpers/`` that a store decorator (``store``, ``store_xarray``,
   ``store_dict``) wraps, and that receives stimuli, texts or paths, has
   ``extraction_fingerprint`` among its key arguments. In unified, no table name
   (``stimuli_identifier``) is a key argument. Old plugin folders are outside the
   scanned folders.
2. Through ``participate``, for every caching wrapper of ``brainscore.model_helpers``
   (the stand-ins of ``tests/brainscore_model/tiny_wrappers.py``) and for the
   ordinary and the temporal vision helper: a variant of a table that is already
   stored (new bytes at an old path, other rows, another order, another column,
   other weights) gives the values of a cold run. For the unified wrappers, the
   same table under two names gives one forward pass and one entry. Every case
   first sees the store write an entry and serve it, so it fails with the store off.
"""
import ast
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import result_caching
import torch
import xarray

from brainscore import BrainScoreModel
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from tests.brainscore_model import tiny_wrappers

pytestmark = pytest.mark.unit

DECORATORS = {'store', 'store_xarray', 'store_dict'}
INPUT_ARGUMENT = re.compile(r'^(stimuli|stimulus_set|inputs|texts?|\w*paths?)$')
TABLE_NAME = re.compile(r'^stimul\w*_identifier$')


# -- 1. the key arguments of every shipped store --------------------------------

def _scanned_folders():
    import brainscore
    import brainscore_vision.model_helpers
    return {'unified': Path(brainscore.__file__).parent, 'vision': Path(brainscore_vision.model_helpers.__file__).parent}


def _decorator_name(decorator):
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return target.attr if isinstance(target, ast.Attribute) else getattr(target, 'id', None)


def _ignored(decorator):
    for keyword in getattr(decorator, 'keywords', ()):
        if keyword.arg == 'identifier_ignore':
            return set(ast.literal_eval(keyword.value))
    return set()


def _stores(folder):
    """``(place, key arguments)`` of every stored function under ``folder`` that receives inputs."""
    for path in sorted(folder.rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if _decorator_name(decorator) not in DECORATORS:
                    continue
                arguments = [a.arg for a in node.args.args + node.args.kwonlyargs if a.arg != 'self']
                if any(INPUT_ARGUMENT.match(argument) for argument in arguments):
                    key = [argument for argument in arguments if argument not in _ignored(decorator)]
                    yield f'{path.relative_to(folder.parent)}:{node.lineno} {node.name}', key


def test_every_shipped_store_of_inputs_keys_on_their_content():
    problems = []
    for repository, folder in _scanned_folders().items():
        stores = list(_stores(folder))
        assert stores, f'no stored function that receives inputs was found under {folder}'
        for place, key in stores:
            if 'extraction_fingerprint' not in key:
                problems.append(f'{place}: its key {key} has no extraction_fingerprint of the inputs and settings')
            if repository == 'unified' and any(TABLE_NAME.match(argument) for argument in key):
                problems.append(f'{place}: its key {key} has a table name; a name is a label, not a key')
    assert not problems, ('A store finds a stored result by the content of its input. Give each store below the '
                          'fingerprint of its ordered inputs and settings (brainscore_core.extraction_cache.'
                          'extraction_fingerprint) as a key argument, and remove the table name from a unified '
                          'key:\n' + '\n'.join(problems))


# -- 2. behavior through participate --------------------------------------------

def _storages(obj):
    """The ``result_caching`` stores of the methods of ``obj`` and of its extractor, by class."""
    for owner in (obj, getattr(obj, '_extractor', None)):
        for cls in type(owner).__mro__ if owner is not None else ():
            for member in vars(cls).values():
                for cell in getattr(member, '__closure__', None) or ():
                    if isinstance(cell.cell_contents, result_caching._Storage):
                        yield cell.cell_contents


@pytest.fixture
def store(monkeypatch, tmp_path):
    """A store folder of this test alone, for the stores of the extractor under test; ``on`` switches it."""
    state = type('Store', (), {'on': True, 'folder': tmp_path / 'store', 'prefixes': ()})()

    def enabled(identifier):
        return state.on and identifier.startswith(state.prefixes)
    monkeypatch.setattr(result_caching, 'is_enabled', enabled)
    monkeypatch.setattr('brainscore_core.extraction_cache.is_enabled', enabled)

    def install(extractor):
        storages = list(_storages(extractor))
        assert storages, f'{type(extractor).__name__} has no store'
        for storage in storages:
            monkeypatch.setattr(storage, '_storage_directory', str(state.folder))
        state.prefixes = ('brainscore.model_helpers.', 'brainscore_vision.model_helpers.activations.')
    state.install = install
    state.entries = lambda: sorted(state.folder.rglob('*.pkl')) if state.folder.exists() else []
    return state


class _Passes:
    """Counts forward passes; a hook with a fixed ``cache_config``, so it does not change the model key."""

    def __init__(self):
        self.count = 0

    def __call__(self, module, args):
        self.count += 1

    def cache_config(self):
        return 'forward pass counter'


def _counted(network):
    passes = _Passes()
    network.register_forward_pre_hook(passes)
    return passes


def _unified(kind, folder):
    wrapper, stimuli, regions, network = tiny_wrappers.FACTORIES[kind](folder)
    passes = _counted(network)
    model = BrainScoreModel(f'guard-{kind}', extractor={tiny_wrappers.MODALITY[kind]: wrapper},
                            region_layer_map=regions)
    return model, wrapper, stimuli, sorted(regions), passes, network


def _images(folder, count=3):
    from PIL import Image
    rng = np.random.default_rng(23)
    paths = []
    for index in range(count):
        paths.append(str(folder / f'picture{index}.png'))
        Image.fromarray(rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)).save(paths[-1])
    return tiny_wrappers._table([f'p{i}' for i in range(count)], 'image_path', paths, 'guard-pictures')


def _vision(folder):
    """The ordinary vision helper (``PytorchWrapper``) on a tiny network."""
    from PIL import Image
    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

    def load(paths):
        return np.stack([np.asarray(Image.open(path), np.float32).transpose(2, 0, 1) / 255 for path in paths])
    network = tiny_wrappers._seeded(3, lambda: torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU(), torch.nn.Conv2d(4, 2, 3), torch.nn.Flatten()))
    passes = _counted(network)
    wrapper = PytorchWrapper(network, load, identifier='guard-vision-helper', batch_size=1)
    model = BrainScoreModel('guard-vision-helper', extractor={'vision': wrapper}, region_layer_map={'IT': '2'})
    return model, wrapper, _images(folder), ['IT'], passes, network


def _temporal(folder):
    """The temporal vision helper on a tiny network; a still image becomes a short video."""
    from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper

    def frames(video):
        return torch.from_numpy(video.to_numpy()[:, :8, :8].astype(np.float32) / 255)
    network = tiny_wrappers._seeded(5, lambda: torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(),
                                                                    torch.nn.Linear(4, 2)))
    passes = _counted(network)
    wrapper = PytorchWrapper('guard-temporal-helper', network, frames, fps=2, img_duration=1000,
                             layer_activation_format={'0': 'THWC', '2': 'THWC'}, batch_size=1, max_workers=1)
    model = BrainScoreModel('guard-temporal-helper', extractor={'vision': wrapper}, region_layer_map={'IT': '2'})
    return model, wrapper, _images(folder), ['IT'], passes, network


SUBJECTS = {**{kind: (lambda folder, kind=kind: _unified(kind, folder)) for kind in tiny_wrappers.KINDS},
            'vision_helper': _vision, 'temporal_helper': _temporal}


def _stored(kind, folder):
    """The subject of ``kind`` if its extractor has a store; else skip (a wrapper that computes each time)."""
    built = SUBJECTS[kind](folder)
    if not list(_storages(built[1])):
        pytest.skip(f'{type(built[1]).__name__} keeps no store')
    return built


def _record(model, stimuli, regions):
    session = stimulus_session(stimuli, [f'neural:{region}' for region in regions])
    model.participate(session)
    return {region: session.collect(f'neural:{region}') for region in regions}


def _variant(stimuli, rows=None, identifier=None, **columns):
    """``stimuli`` with ``rows`` (positions) and extra ``columns``, under ``identifier`` (default: its own)."""
    rows = list(range(len(stimuli))) if rows is None else rows
    table = StimulusSet({column: [stimuli[column].values[i] for i in rows] for column in stimuli.columns})
    for column, values in columns.items():
        table[column] = values
    table.identifier = stimuli.identifier if identifier is None else identifier
    paths = getattr(stimuli, 'stimulus_paths', None)
    if paths:
        table.stimulus_paths = {stimulus_id: paths[stimulus_id] for stimulus_id in table['stimulus_id']}
    return table


def _input_column(stimuli):
    return next(column for column in stimuli.columns if column != 'stimulus_id')


def _replace_first(stimuli, network):
    """New content for the first row at its old place: the bytes of another file, or another text."""
    column = _input_column(stimuli)
    first, other = stimuli[column].values[0], stimuli[column].values[1]
    if os.path.isfile(str(first)):
        shutil.copyfile(other, first)
        return stimuli
    return _variant(stimuli, **{column: [other] + list(stimuli[column].values[1:])})


def _other_weights(stimuli, network):
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.add_(0.25)
    return stimuli


CHANGES = {
    'new_bytes_at_an_old_path': _replace_first,
    'other_rows': lambda stimuli, network: _variant(stimuli, rows=[0, 2]),
    'another_order': lambda stimuli, network: _variant(stimuli, rows=list(range(len(stimuli)))[::-1]),
    'another_column': lambda stimuli, network: _variant(stimuli, extra=[f'x{i}' for i in range(len(stimuli))]),
    'other_weights': _other_weights,
}


def _assert_same(actual, expected):
    """Equal values (bit for bit), dims, coordinates and attributes in every region."""
    for region in expected:
        assert actual[region].values.tobytes() == expected[region].values.tobytes(), region
        xarray.testing.assert_identical(actual[region], expected[region])


def _stored_and_served(store, model, stimuli, regions, passes):
    """Record ``stimuli`` cold into the store, then see the store serve it with no forward pass."""
    first = _record(model, stimuli, regions)
    assert store.entries(), 'the store wrote no entry: the check would prove nothing'
    before = passes.count
    _assert_same(_record(model, stimuli, regions), first)
    assert passes.count == before, 'the store did not serve a stored table'


@pytest.mark.parametrize('change', sorted(CHANGES))
@pytest.mark.parametrize('kind', sorted(SUBJECTS))
def test_a_changed_table_under_its_old_name_gives_the_cold_values(kind, change, store, tmp_path, monkeypatch):
    pytest.importorskip('cv2')
    monkeypatch.chdir(tmp_path)
    model, extractor, stimuli, regions, passes, network = _stored(kind, tmp_path)
    store.install(extractor)
    _stored_and_served(store, model, stimuli, regions, passes)
    changed = CHANGES[change](stimuli, network)
    warm = _record(model, changed, regions)
    store.on = False
    _assert_same(warm, _record(model, changed, regions))


@pytest.mark.parametrize('kind', tiny_wrappers.KINDS)
def test_the_same_table_under_two_names_takes_one_forward_pass(kind, store, tmp_path, monkeypatch):
    pytest.importorskip('cv2')
    monkeypatch.chdir(tmp_path)
    model, extractor, stimuli, regions, passes, network = _stored(kind, tmp_path)
    store.install(extractor)
    _stored_and_served(store, model, stimuli, regions, passes)
    entries, before = store.entries(), passes.count
    renamed = _record(model, _variant(stimuli, identifier='another-name'), regions)
    assert passes.count == before, 'a second name of the same table ran the network again'
    assert store.entries() == entries, 'a second name of the same table wrote a second entry'
    store.on = False
    _assert_same(renamed, _record(model, _variant(stimuli, identifier='another-name'), regions))


def test_every_caching_wrapper_has_a_stand_in(tmp_path):
    """Discovery: each subclass of the shared caching extractor in ``brainscore.model_helpers`` has a stand-in."""
    import importlib
    import pkgutil
    import brainscore.model_helpers as helpers
    from brainscore.model_helpers._extractors import CachingExtractor
    found = set()
    for info in pkgutil.walk_packages(helpers.__path__, helpers.__name__ + '.'):
        module = importlib.import_module(info.name)
        found |= {member for member in vars(module).values() if isinstance(member, type)
                  and issubclass(member, CachingExtractor) and member is not CachingExtractor
                  and member.__module__ == module.__name__}
    pytest.importorskip('cv2')
    covered = {type(tiny_wrappers.FACTORIES[kind](tmp_path)[0]) for kind in tiny_wrappers.KINDS}
    missing = sorted(cls.__qualname__ for cls in found - covered)
    assert not missing, (f'{missing} cache activations and have no stand-in, so the content guard does not run '
                         f'them. Add a factory for each to tests/brainscore_model/tiny_wrappers.py (FACTORIES, '
                         f'MODALITY, KINDS).')
