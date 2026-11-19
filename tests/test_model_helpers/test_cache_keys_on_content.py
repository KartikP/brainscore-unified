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
   ordinary and the temporal vision helper (an extractor with no store must
   compute each time): a variant of a table that is already
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


def _unified(kind, folder, regions_out=None):
    wrapper, stimuli, regions, network = tiny_wrappers.FACTORIES[kind](folder)
    passes = _counted(network)
    model = BrainScoreModel(f'guard-{kind}', extractor={tiny_wrappers.MODALITY[kind]: wrapper},
                            region_layer_map=regions)
    if regions_out is not None:
        regions_out.update({region: selector.layer_path for region, selector in regions.items()})
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


def _built(kind, folder):
    """The subject of ``kind``, and whether its extractor keeps a store; a skip when its stand-in needs OpenCV.

    Whether an extractor keeps a store is stated by what it is, not found from its code: a subclass of
    the shared caching extractor and the two vision helpers store; any other extractor computes each time.
    """
    from brainscore.model_helpers._extractors import CachingExtractor
    try:
        built = SUBJECTS[kind](folder)
    except ModuleNotFoundError as error:
        if error.name != 'cv2':
            raise
        pytest.skip('the stand-in writes its clips with OpenCV')
    stores = isinstance(built[1], CachingExtractor) or kind in ('vision_helper', 'temporal_helper')
    assert stores == bool(list(_storages(built[1]))), (
        f'{type(built[1]).__qualname__} {"has lost" if stores else "has gained"} its activation store')
    return built, stores


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


def _computes_each_time(store, model, stimuli, regions, passes):
    """An extractor with no store runs the network for every table and writes no entry."""
    first, before = _record(model, stimuli, regions), passes.count
    _assert_same(_record(model, stimuli, regions), first)
    assert passes.count > before and not store.entries(), 'an extractor with no store served a table'


def _stored_and_served(store, model, stimuli, regions, passes):
    """Record ``stimuli`` cold into the store, then see the store serve it with no forward pass.

    A wrapper whose first pass sets an attribute of its key (the audio step) computes the
    table once more under that state; the next call is served.
    """
    first = _record(model, stimuli, regions)
    assert store.entries(), 'the store wrote no entry: the check would prove nothing'
    for _ in range(2):
        before = passes.count
        _assert_same(_record(model, stimuli, regions), first)
        if passes.count == before:
            return
    pytest.fail('the store did not serve a stored table')


@pytest.mark.parametrize('change', sorted(CHANGES))
@pytest.mark.parametrize('kind', sorted(SUBJECTS))
def test_a_changed_table_under_its_old_name_gives_the_cold_values(kind, change, store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (model, extractor, stimuli, regions, passes, network), stored = _built(kind, tmp_path)
    if not stored:
        return _computes_each_time(store, model, stimuli, regions, passes)
    store.install(extractor)
    _stored_and_served(store, model, stimuli, regions, passes)
    changed = CHANGES[change](stimuli, network)
    warm = _record(model, changed, regions)
    store.on = False
    _assert_same(warm, _record(model, changed, regions))


@pytest.mark.parametrize('kind', tiny_wrappers.KINDS)
def test_the_same_table_under_two_names_takes_one_forward_pass(kind, store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (model, extractor, stimuli, regions, passes, network), stored = _built(kind, tmp_path)
    if not stored:
        return _computes_each_time(store, model, stimuli, regions, passes)
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
    try:
        covered = {type(tiny_wrappers.FACTORIES[kind](tmp_path)[0]) for kind in tiny_wrappers.KINDS}
    except ModuleNotFoundError as error:
        pytest.fail(f'The stand-ins of tests/brainscore_model/tiny_wrappers.py need {error.name}; install it '
                    f'(opencv-python for cv2): this check must not be skipped.')
    missing = sorted(cls.__qualname__ for cls in found - covered)
    assert not missing, (f'{missing} cache activations and have no stand-in, so the content guard does not run '
                         f'them. Add a factory for each to tests/brainscore_model/tiny_wrappers.py (FACTORIES, '
                         f'MODALITY, KINDS).')


# -- 3. everything that decides a stored value is in its key, or the store is off ---

def _zero_after(module, args, *output):
    """One hook for both places: it zeroes a module's output, and does nothing before the module."""
    if output:
        return output[0] * 0


def _ready(kind, store, folder, monkeypatch, layers=None):
    """The unified subject of ``kind`` with its store installed, or a skip when its stand-in needs OpenCV."""
    monkeypatch.chdir(folder)
    try:
        built = _unified(kind, folder, layers)
    except ModuleNotFoundError as error:
        if error.name != 'cv2':
            raise
        pytest.skip('the stand-in writes its clips with OpenCV')
    store.install(built[1])
    return built


@pytest.mark.parametrize('kind', [kind for kind in tiny_wrappers.KINDS if kind != 'vlm_video'])
def test_a_hook_moved_from_before_a_layer_to_after_it_gives_the_cold_values(kind, store, tmp_path, monkeypatch):
    layers = {}
    model, wrapper, stimuli, regions, passes, network = _ready(kind, store, tmp_path, monkeypatch, layers)
    module = network.get_submodule(layers['IT'])
    before = module.register_forward_pre_hook(_zero_after)
    _stored_and_served(store, model, stimuli, regions, passes)
    before.remove()
    after = module.register_forward_hook(_zero_after)
    warm = _record(model, stimuli, regions)
    store.on = False
    cold = _record(model, stimuli, regions)
    after.remove()
    assert not np.any(cold['IT'].values)
    _assert_same(warm, cold)



def test_the_time_bins_of_a_reader_do_not_depend_on_what_another_reader_saw(store, tmp_path, monkeypatch):
    """The audio step is measured on a wrapper's first pass: a writer that first saw a short clip
    keeps another step than a fresh reader, so the step is a part of the key."""
    model, writer, stimuli, regions, passes, network = _ready('audio', store, tmp_path, monkeypatch)
    short = tmp_path / 'short.npy'
    np.save(short, np.ones(31, dtype=np.float32))
    writer([str(short)], ['head'])                           # a first pass that sets another step
    _stored_and_served(store, model, stimuli, regions, passes)
    warm = _record(_audio_reader(tmp_path), stimuli, regions)
    store.on = False
    _assert_same(warm, _record(_audio_reader(tmp_path), stimuli, regions))


def _audio_reader(folder):
    """A fresh audio subject with the weights, settings and hooks of the writer."""
    reader, _, regions, network = tiny_wrappers.audio(folder)
    _counted(network)
    return BrainScoreModel('guard-audio-reader', extractor={'audio': reader}, region_layer_map=regions)


def _others(value, owner):
    """Other values of the type of ``value``, found from the value itself and the class of ``owner``.

    A number grows, a flag turns, a string becomes another member of a tuple of strings in the class
    (the allowed values of a setting); a function becomes a function of other code with the same result.
    Anything else gives none: the guard cannot vary it.
    """
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value + 1]
    if isinstance(value, float):
        return [value * 2]
    if isinstance(value, str):
        return [other for member in vars(type(owner)).values() if isinstance(member, tuple) and value in member
                for other in member if other != value]
    if callable(value) and getattr(value, '__code__', None) is not None:
        return [lambda *args, _value=value, **kwargs: _value(*args, **kwargs)]
    return []


def _arguments(kind, folder):
    """The constructor arguments the stand-in of ``kind`` gives its wrapper, read from one construction."""
    import inspect
    cls = type(tiny_wrappers.FACTORIES[kind](folder)[0])
    seen = {}
    original = cls.__init__

    def record(self, *args, **kwargs):
        bound = inspect.signature(original).bind(self, *args, **kwargs)
        bound.apply_defaults()
        seen.update(bound.arguments)
        original(self, *args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cls, '__init__', record)
        tiny_wrappers.FACTORIES[kind](folder)
    seen.pop('self')
    return cls, seen


def _changed(kind, folder, name, value, monkeypatch):
    """The subject of ``kind`` whose wrapper got ``value`` for the constructor argument ``name``."""
    cls = type(tiny_wrappers.FACTORIES[kind](folder)[0])
    original = cls.__init__

    def change(self, *args, **kwargs):
        import inspect
        bound = inspect.signature(original).bind(self, *args, **kwargs)
        bound.arguments[name] = value
        original(*bound.args, **bound.kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cls, '__init__', change)
        return _unified(kind, folder)


def _misses(store, model, stimuli, regions, passes):
    """True when ``model`` computes ``stimuli`` (another key or no store), not served by the store."""
    before = passes.count
    _record(model, stimuli, regions)
    return passes.count > before


class _Opaque:
    """An object with no description: a key cannot hold it."""


STORED_KINDS = [kind for kind in tiny_wrappers.KINDS if kind != 'vlm_video']


@pytest.mark.parametrize('kind', STORED_KINDS)
def test_every_constructor_argument_is_in_the_key_or_turns_the_store_off(kind, store, tmp_path, monkeypatch):
    """Each argument of the wrapper's constructor (from its signature): another value is another key, or
    no store. The identifier is a label (two names share one entry, the test above). An argument the guard
    cannot vary must be kept on the wrapper as ``<name>`` or ``_<name>``: every attribute of a wrapper is
    in its key (the next test), so a new argument is covered when it is kept, and named here when not."""
    model, wrapper, stimuli, regions, passes, network = _ready(kind, store, tmp_path, monkeypatch)
    _stored_and_served(store, model, stimuli, regions, passes)
    cls, arguments = _arguments(kind, tmp_path)
    escaped = []
    for name, value in arguments.items():
        if name == 'identifier':
            continue
        others = _others(value, wrapper)
        if others:
            changed = _changed(kind, tmp_path, name, others[0], monkeypatch)
            store.install(changed[1])
            if not _misses(store, changed[0], stimuli, regions, changed[4]):
                escaped.append(f'{name}={others[0]!r} was served the entry of {name}={value!r}')
        elif not ({name, f'_{name}'} & set(vars(wrapper))):
            escaped.append(f'{name} cannot be varied here and is kept in no attribute of the wrapper')
    assert not escaped, (f'{cls.__qualname__}: an argument that can change a stored value is not in the key. '
                         f'Keep it as an attribute of the wrapper (every attribute is in the key): {escaped}')


@pytest.mark.parametrize('kind', STORED_KINDS)
def test_every_attribute_of_a_wrapper_is_in_the_key_or_turns_the_store_off(kind, store, tmp_path, monkeypatch):
    """Each attribute of the wrapper after one extraction (so an attribute set by the first pass is seen),
    apart from the identifier label, the model (its weights and hooks) and the input device: another value
    is another key or no store. An attribute the guard cannot vary gets an object no key can describe,
    which must turn the store off (the extraction then computes, or fails on the object)."""
    model, wrapper, stimuli, regions, passes, network = _ready(kind, store, tmp_path, monkeypatch)
    _stored_and_served(store, model, stimuli, regions, passes)
    escaped = []
    for name, value in list(vars(wrapper).items()):
        if name in ('_identifier', '_model', '_device'):
            continue
        other = (_others(value, wrapper) or [_Opaque()])[0]
        setattr(wrapper, name, other)
        try:
            served = not _misses(store, model, stimuli, regions, passes)
        except Exception:                                   # noqa: BLE001 - it computed, with a value it cannot use
            served = False
        if served:
            escaped.append(f'{name}: {other!r} was served the entry of {value!r}')
        setattr(wrapper, name, value)
    assert not escaped, f'{type(wrapper).__qualname__}: an attribute is not in the key: {escaped}'
