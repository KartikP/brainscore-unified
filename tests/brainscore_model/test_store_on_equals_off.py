"""A subject's neural output is the same with the activation store off and on (cold, warm), bit for bit.

An extractor with its own store returns stored activations in another memory
layout than computed ones (the store selects layers with an integer index), and
a reduction rounds by the layout. The model class puts every extractor result in
C order where it receives it (``dispatch.call_extractor``). The extractors here
are the legacy vision ``PytorchWrapper`` (with and without a ``LayerPCA`` hook)
and a custom callable with a store of its own. The store is a temporary folder
that the test removes.
"""
import os
import tempfile

import numpy as np
import pytest
import result_caching
import torch
import xarray
from PIL import Image

from brainscore import BrainScoreModel
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


class _CustomStoredExtractor:
    """A callable extractor with a ``result_caching`` store of its own; values over (image, time bin, unit)."""

    def __init__(self):
        torch.manual_seed(1)
        self.weights = torch.randn(4 * 4 * 3, 6).numpy() * 0.37

    def __call__(self, stimuli, layers, **kwargs):
        return self._stored(identifier='custom-stored', layers=list(layers),
                            stimuli_identifier=stimuli.identifier, paths=list(stimuli['image_path']))

    @result_caching.store_xarray(identifier_ignore=['paths', 'layers'], combine_fields={'layers': 'layer'})
    def _stored(self, identifier, layers, stimuli_identifier, paths):
        pixels = np.stack([np.asarray(Image.open(path), np.float32)[:4, :4].reshape(-1) / 255 for path in paths])
        steps = np.stack([np.sin(pixels @ self.weights * (step + 1)) for step in range(5)], axis=1)
        values = np.concatenate([steps * (index + 1.3) for index in range(len(layers))], axis=2).astype(np.float32)
        return NeuroidAssembly(values, dims=['presentation', 'time_bin', 'neuroid'], coords={
            'stimulus_path': ('presentation', paths), 'time_bin_id': ('time_bin', list(range(5))),
            'layer': ('neuroid', [layer for layer in layers for _ in range(6)]),
            'neuroid_id': ('neuroid', [f'{layer}.{unit}' for layer in layers for unit in range(6)])})


def _storages(*methods):
    for method in methods:
        for cell in method.__func__.__closure__ or ():
            if isinstance(cell.cell_contents, result_caching._Storage):
                yield cell.cell_contents


@pytest.fixture
def store(monkeypatch, tmp_path):
    folder = tempfile.TemporaryDirectory()
    state = {'on': False}
    patterns = ('ActivationsExtractorHelper', 'LayerPCA', '_CustomStoredExtractor')

    def enable(identifier):
        return state['on'] and any(pattern in identifier for pattern in patterns)
    monkeypatch.setattr(result_caching, 'is_enabled', enable)
    monkeypatch.setattr('brainscore_core.extraction_cache.is_enabled', enable)

    def install(*methods):
        for storage in _storages(*methods):
            monkeypatch.setattr(storage, '_storage_directory', folder.name)
    yield state, install, folder.name
    folder.cleanup()


def _images(folder, prefix, count, seed):
    rng = np.random.default_rng(seed)
    paths = []
    for index in range(count):
        paths.append(str(folder / f'{prefix}{index}.png'))
        Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(paths[-1])
    return paths


def _table(paths):
    table = StimulusSet({'stimulus_id': [f's{index}' for index in range(len(paths))], 'image_path': paths})
    table.identifier = 'store-on-off'
    table.stimulus_paths = dict(zip(table['stimulus_id'], paths))
    return table


def _record(subject, table):
    from brainscore_core.streaming_helpers import stimulus_session
    session = stimulus_session(table, ['neural:IT'])
    subject.participate(session)
    return session.collect('neural:IT')


def _legacy(tmp_path, monkeypatch, install, pca):
    from brainscore_vision.model_helpers.activations import pca as pca_module
    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    fit_images = _images(tmp_path, 'fit', 12, seed=3)
    monkeypatch.setattr(pca_module, '_get_imagenet_val', lambda num_images: fit_images)

    def load(paths):
        return np.stack([np.asarray(Image.open(path), np.float32).transpose(2, 0, 1) / 255 for path in paths])
    torch.manual_seed(0)
    network = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU(), torch.nn.Conv2d(4, 4, 3),
                                  torch.nn.Flatten())
    wrapper = PytorchWrapper(network, load, identifier='store-on-off-net', batch_size=2)
    install(wrapper._extractor._from_paths_stored)
    if pca:
        pca_module.LayerPCA.hook(wrapper, n_components=3)
        install(next(iter(wrapper._extractor._batch_activations_hooks.values()))._pcas)
    return BrainScoreModel('store-on-off', extractor={'vision': wrapper}, region_layer_map={'IT': '2'})


def _custom(install):
    extractor = _CustomStoredExtractor()
    install(extractor._stored)
    return BrainScoreModel('store-on-off-custom', extractor={'vision': extractor}, region_layer_map={'IT': 'block'})


@pytest.mark.parametrize('case', ['legacy', 'legacy-pca', 'custom'])
def test_store_off_cold_and_warm_give_the_same_neural_output(case, store, tmp_path, monkeypatch):
    state, install, folder = store
    subject = _custom(install) if case == 'custom' else _legacy(tmp_path, monkeypatch, install, case == 'legacy-pca')
    table = _table(_images(tmp_path, 'stim', 5, seed=4))
    state['on'] = False
    off = _record(subject, table)
    state['on'] = True
    cold, warm = _record(subject, table), _record(subject, table)
    assert any(name.endswith('.pkl') for _, _, names in os.walk(folder) for name in names)    # the store was used
    for result in (cold, warm):
        xarray.testing.assert_identical(result.reset_index(list(result.indexes)), off.reset_index(list(off.indexes)))
        assert result.values.strides == off.values.strides and result.values.flags['C_CONTIGUOUS']
        for dim in off.dims:                                  # over images, over time (custom) and over units
            np.testing.assert_array_equal(result.astype(np.float32).sum(dim).values,
                                          off.astype(np.float32).sum(dim).values)
