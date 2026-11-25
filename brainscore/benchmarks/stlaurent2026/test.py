"""The THINGS benchmarks on a small made-up packaged folder (no download, no real model)."""
import hashlib
import zlib

import numpy as np
import pandas as pd
import pytest

import brainscore
from brainscore import BrainScoreModel
from brainscore.benchmarks.stlaurent2026 import benchmark as things_benchmark
from brainscore.benchmarks.stlaurent2026.benchmark import TEST_CATEGORIES, StLaurent2026Things
from brainscore.data.cneuromod import data as things_data
from brainscore.data.cneuromod.data_packaging import shared
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

pytestmark = pytest.mark.unit

N_LATENT = 6
CATEGORIES = sorted(TEST_CATEGORIES)[:6] + [f'made_up_{i:02d}' for i in range(24)]
IMAGES = [f'{category}_{k:02d}a' for category in CATEGORIES for k in range(6)]
LATENT = dict(zip(IMAGES, np.random.default_rng(0).standard_normal((len(IMAGES), N_LATENT))))


class _Extractor:
    """The made-up model: each layer answers a fixed linear map of the image's latent features."""

    identifier = 'made-up-things-extractor'

    def __call__(self, stimuli, layers):
        latent = np.array([LATENT[i] for i in stimuli['stimulus_id']])
        blocks = [latent @ np.random.default_rng(zlib.crc32(layer.encode())).standard_normal((N_LATENT, 12))
                  for layer in layers]
        return NeuroidAssembly(np.concatenate(blocks, axis=1), dims=['presentation', 'neuroid'], coords={
            **{column: ('presentation', list(stimuli[column])) for column in stimuli.columns},
            'layer': ('neuroid', [layer for layer in layers for _ in range(12)]),
            'neuroid_id': ('neuroid', [f'{layer}.{unit}' for layer in layers for unit in range(12)])})


def _region(rng, name, atlas, n_voxels, signal=True, test_only=False):
    weights = rng.standard_normal((N_LATENT, n_voxels))
    signal_of = {image: (LATENT[image] @ weights if signal else np.zeros(n_voxels)) for image in IMAGES}
    if test_only:      # signal on the test images only: a selection on the training images must not take it
        signal_of = {image: s * (image.rsplit('_', 1)[0] in TEST_CATEGORIES) for image, s in signal_of.items()}
    return name, atlas, n_voxels, signal_of


def _package(root, participant, regions, seed):
    """Two sessions of ``trials.nc``; each image three times, every trial answered."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.repeat(IMAGES, 3))
    neuroid = pd.concat([pd.DataFrame({'neuroid_id': [f'{atlas}.{name}.{v}' for v in range(n)], 'region': name,
                                       'atlas': atlas, 'voxel': np.arange(n), 'i': 0, 'j': 0, 'k': 0})
                         for name, atlas, n, _ in regions], ignore_index=True)
    values = np.concatenate([np.array([signal[image] for image in order]) for *_, signal in regions], axis=1)
    values = values + 0.6 * rng.standard_normal(values.shape)
    for session, rows in enumerate(np.array_split(np.arange(len(order)), 2), start=1):
        folder = root / participant / f'ses-{session:02d}'
        presentation = pd.DataFrame({'stimulus_id': order[rows], 'response': 1.0})
        shared.write_assembly(values[rows], presentation, neuroid, folder / 'trials.nc')
        presentation[['stimulus_id']].drop_duplicates().to_csv(folder / 'stimuli.csv', index=False)


@pytest.fixture
def packaged(tmp_path, monkeypatch):
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path / 'home'))
    for image in IMAGES:
        path = tmp_path / 'home' / 'things-images' / image.rsplit('_', 1)[0] / f'{image}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image.encode())
    record = tmp_path / 'sha1.csv'
    pd.DataFrame({'stimulus_id': IMAGES, 'sha1': [hashlib.sha1(i.encode()).hexdigest() for i in IMAGES]}
                 ).to_csv(record, index=False)
    monkeypatch.setattr(things_data, 'IMAGE_SHA1', record)
    rng = np.random.default_rng(2)
    root = tmp_path / 'packaged'
    _package(root, 'sub-01', [_region(rng, 'V1', 'retinoVisionNpythy', 60), _region(rng, 'hV4', 'retinoVisionNpythy', 60),
                              _region(rng, 'FFA', 'fLocVisionTask', 60),
                              _region(rng, 'EBA', 'fLocVisionTask', 60, signal=False),
                              _region(rng, 'PPA', 'fLocVisionTask', 60, test_only=True),
                              _region(rng, 'cortex', 'aparcaseg', 80)], seed=3)
    _package(root, 'sub-02', [_region(rng, 'V1', 'retinoVisionNpythy', 60), _region(rng, 'hV4', 'retinoVisionNpythy', 60),
                              _region(rng, 'FFA', 'fLocVisionTask', 20), _region(rng, 'EBA', 'fLocVisionTask', 60),
                              _region(rng, 'PPA', 'fLocVisionTask', 60), _region(rng, 'cortex', 'aparcaseg', 80)],
             seed=4)
    return root


def _model(regions=('V1', 'V4', 'FFA', 'EBA', 'PPA'), cortex=('early', 'late')):
    from brainscore.selection import CompositeSelector
    layers = {region: 'early' for region in regions}
    layers['cortex'] = CompositeSelector(tuple((layer, None) for layer in cortex))
    return BrainScoreModel('made-up-things', region_layer_map=layers, extractor={'vision': _Extractor()})


def _benchmark(region, root, participants=('sub-01', 'sub-02')):
    return StLaurent2026Things(region, participants, identifier=f'StLaurent2026-fMRI-things-{region}', root=root)


def test_a_model_whose_layer_carries_the_signal_scores_near_its_ceiling(packaged):
    score = brainscore.score(_model(), _benchmark('V1', packaged), check_mem=False)
    assert score.attrs['participants'] == ['sub-01', 'sub-02']
    assert 0.8 < float(score) < 1.2 and score.attrs['normalized']
    assert float(score.attrs['raw']) > 0.5 and float(score.attrs['ceiling']) > 0.5
    facts = score.attrs['per_participant']['sub-01']
    assert facts['n_images_scored'] == 36 and facts['n_images_fitted'] == 144
    assert facts['atlas'] == 'retinoVisionNpythy' and facts['n_voxels_selected'] >= 50


def test_v4_reads_the_hv4_mask(packaged):
    score = brainscore.score(_model(), _benchmark('V4', packaged), check_mem=False)
    assert score.attrs['per_participant']['sub-02']['n_voxels_in_mask'] == 60
    assert 'hV4' in score.attrs['region_note']


def test_a_participant_with_too_few_selected_voxels_does_not_count_and_is_stated(packaged):
    score = brainscore.score(_model(), _benchmark('FFA', packaged), check_mem=False)
    assert score.attrs['participants'] == ['sub-01']
    assert 'excluded' in score.attrs['per_participant']['sub-02']


def test_no_participant_with_enough_voxels_is_an_error_not_a_score(packaged):
    with pytest.raises(ValueError, match='nothing to score'):
        brainscore.score(_model(), _benchmark('EBA', packaged, participants=('sub-01',)), check_mem=False)


def test_the_voxels_are_selected_on_the_training_images_only(packaged):
    """Voxels with signal on the test images only have no noise ceiling on the training images."""
    score = brainscore.score(_model(), _benchmark('PPA', packaged), check_mem=False)
    assert score.attrs['per_participant']['sub-01']['n_voxels_selected'] < 50
    assert score.attrs['participants'] == ['sub-02']


def test_no_test_category_reaches_the_fit(packaged, monkeypatch):
    seen = []
    real = things_benchmark.cross_validated_r

    def spy(features, Y, folds, fit, *, groups=None, **kwargs):
        seen.append((folds, groups))
        return real(features, Y, folds, fit, groups=groups, **kwargs)
    monkeypatch.setattr(things_benchmark, 'cross_validated_r', spy)
    brainscore.score(_model(), _benchmark('V1', packaged), check_mem=False)
    assert seen
    for folds, groups in seen:
        for train, test in folds:
            assert not set(groups[train]) & TEST_CATEGORIES
            assert set(groups[test]) <= TEST_CATEGORIES


def test_an_image_with_other_bytes_than_the_things_database_is_refused_by_name(packaged, tmp_path):
    changed = tmp_path / 'home' / 'things-images' / CATEGORIES[3] / f'{IMAGES[20]}.jpg'
    changed.write_bytes(b'a resized copy')
    with pytest.raises(ValueError, match=IMAGES[20]):
        brainscore.load_stimulus_set('cneuromod-things', root=packaged)


def test_the_recorded_sha1_lists_the_4320_images_of_the_sessions():
    record = pd.read_csv(things_data.IMAGE_SHA1)
    assert len(record) == 4320 and record['stimulus_id'].is_unique and record['sha1'].str.len().eq(40).all()


def test_the_stored_split_holds_144_of_720_categories():
    assert len(TEST_CATEGORIES) == 144


def test_a_model_without_a_layer_for_the_region_is_not_applicable_before_any_data_is_read(tmp_path):
    benchmark = _benchmark('FFA', tmp_path / 'nothing-packaged-here')
    score = brainscore.score(_model(regions=('V1',)), benchmark, check_mem=False)
    assert np.isnan(float(score))
    assert 'neural:FFA' in str(score.attrs)


def test_the_cortex_form_fits_one_band_per_layer_of_the_cortex_selector(packaged):
    score = brainscore.score(_model(), _benchmark('cortex', packaged), check_mem=False)
    assert score.attrs['bands'] == ['early', 'late']
    assert score.attrs['per_participant']['sub-01']['components_per_band'] == {'early': 12, 'late': 12}
    assert float(score) > 0.8 and score.attrs['per_participant']['sub-01']['atlas'] == 'aparcaseg'


def test_the_cortex_form_refuses_more_than_12_layers(packaged):
    model = _model(cortex=tuple(f'layer{i}' for i in range(13)))
    with pytest.raises(ValueError, match='at most 12'):
        brainscore.score(model, _benchmark('cortex', packaged), check_mem=False)


def test_every_region_is_registered_with_its_request():
    for region in ('V1', 'V2', 'V3', 'V4', 'FFA', 'OFA', 'pSTS', 'PPA', 'OPA', 'MPA', 'EBA', 'cortex'):
        benchmark = brainscore.load_benchmark(f'StLaurent2026-fMRI-things-{region}')
        assert benchmark.requested_output_channels == {f'neural:{region}'}
        assert benchmark.required_input_channels == {'vision:image'}
