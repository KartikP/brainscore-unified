"""The mario benchmark on a small made-up packaged folder (no download, no real model)."""
import zlib

import numpy as np
import pandas as pd
import pytest

import brainscore
from brainscore import BrainScoreModel
from brainscore.benchmarks.paugam2025 import benchmark as mario_benchmark
from brainscore.benchmarks.paugam2025.benchmark import TEST_RUNS, Paugam2025Mario, Paugam2025MarioButtons
from brainscore.data.cneuromod.data_packaging import mario, shared
from brainscore.temporal import hrf_convolve
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

pytestmark = pytest.mark.unit

N_LATENT, N_TRS, N_PARCELS = 6, 60, 30
RUNS = [(1, run) for run in range(1, 9)]                      # session 1, runs 1-8
MADE_UP_TEST = frozenset({'sub-01_ses-001_run-7', 'sub-01_ses-001_run-8',
                          'sub-02_ses-001_run-7', 'sub-02_ses-001_run-8'})
LATENT = {}


class _Extractor:
    """The made-up model: each layer answers a fixed linear map of the TR's latent features."""

    identifier = 'made-up-mario-extractor'

    def __call__(self, stimuli, layers):
        latent = np.array([LATENT[i] for i in stimuli['stimulus_id']])
        blocks = [latent @ np.random.default_rng(zlib.crc32(layer.encode())).standard_normal((N_LATENT, 12))
                  for layer in layers]
        return NeuroidAssembly(np.concatenate(blocks, axis=1), dims=['presentation', 'neuroid'], coords={
            **{column: ('presentation', list(stimuli[column])) for column in stimuli.columns},
            'layer': ('neuroid', [layer for layer in layers for _ in range(12)]),
            'neuroid_id': ('neuroid', [f'{layer}.{unit}' for layer in layers for unit in range(12)])})


def _package(root, participant, seed):
    """One session of 8 runs, 60 TRs each; TRs 5-54 show a play (media rows), the others none."""
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal((N_LATENT, N_PARCELS))
    presentations, values, media = [], [], []
    for session, run in RUNS:
        latent = np.zeros((N_TRS, N_LATENT))
        for tr in range(5, 55):
            name = mario.tr_identifier(participant, session, run, tr)
            LATENT[name] = latent[tr] = rng.standard_normal(N_LATENT)
            folder = root / participant / 'media' / 'ses-001'
            folder.mkdir(parents=True, exist_ok=True)
            for suffix in ('mp4', 'png', 'wav'):
                (folder / f'{name}.{suffix}').write_bytes(b'')
            media.append({'stimulus_id': name, 'session': session, 'run': run, 'tr': tr,
                          'repetition_id': f'rep-{run}', 'frames': 89, 'video_path': f'{participant}/media/ses-001/{name}.mp4',
                          'image_path': f'{participant}/media/ses-001/{name}.png',
                          'audio_path': f'{participant}/media/ses-001/{name}.wav'})
        values.append(hrf_convolve(latent @ weights, 1 / 1.49) + 0.3 * rng.standard_normal((N_TRS, N_PARCELS)))
        presentations.append(pd.DataFrame({'participant': participant, 'session': session, 'run': run,
                                           'tr': np.arange(N_TRS), 'stimulus_id': '',
                                           # the player holds RIGHT when latent 0 is high, A when latent 1 is
                                           'button_RIGHT': 1.2 * (latent[:, 0] > 0), 'button_A': 1.0 * (latent[:, 1] > 0.5),
                                           **{f'button_{b}': 0.0 for b in ('B', 'LEFT', 'UP', 'DOWN')}}))
    neuroid = pd.DataFrame({'parcel': np.arange(1, N_PARCELS + 1), 'parcel_name': [f'P{i}' for i in range(N_PARCELS)]})
    shared.write_assembly(np.concatenate(values), pd.concat(presentations, ignore_index=True), neuroid,
                          root / participant / 'ses-01' / 'timeseries_cneuromod2026.nc')
    shared.write_csv(pd.DataFrame(media), root / participant / 'media' / 'media.csv')


@pytest.fixture
def packaged(tmp_path, monkeypatch):
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path / 'home'))
    monkeypatch.setattr(mario_benchmark, 'TEST_RUNS', MADE_UP_TEST)
    root = tmp_path / 'packaged'
    _package(root, 'sub-01', seed=3)
    _package(root, 'sub-02', seed=4)
    return root


def _model(key='vision', cortex=('early', 'late')):
    from brainscore.selection import CompositeSelector
    layers = {'cortex': CompositeSelector(tuple((layer, None) for layer in cortex))}
    return BrainScoreModel('made-up-mario', region_layer_map=layers, extractor={key: _Extractor()},
                           behavioral_readout_layer='early', answer_owners={'behavior:choice': ('readout', 'argmax')})


def _benchmark(root):
    return Paugam2025Mario(('sub-01', 'sub-02'), identifier='Paugam2025-fMRI-mario-cortex', root=root)


@pytest.mark.parametrize('key', ['vision', 'video'])
def test_a_model_whose_layers_carry_the_signal_of_the_play_predicts_the_held_out_runs(packaged, key):
    score = brainscore.score(_model(key), _benchmark(packaged), check_mem=False)
    assert score.attrs['participants'] == ['sub-01', 'sub-02']
    assert float(score) > 0.6 and not score.attrs['normalized']
    facts = score.attrs['per_participant']['sub-01']
    assert facts['n_test_runs'] == 2 and facts['n_trs_scored'] == 2 * N_TRS
    assert facts['n_trs_with_play'] == 8 * 50 and facts['components_per_band'] == {'early': 12, 'late': 12}
    assert len(facts['per_parcel_r']) == N_PARCELS
    assert score.attrs['references']['task_regressors']['mean'] < score.attrs['references']['random-vit-b-32']['mean']


def test_no_test_run_reaches_the_fit(packaged, monkeypatch):
    seen = []
    real = mario_benchmark.cross_validated_r

    def spy(features, Y, folds, fit, *, groups=None, **kwargs):
        seen.append((folds, groups))
        return real(features, Y, folds, fit, groups=groups, **kwargs)
    monkeypatch.setattr(mario_benchmark, 'cross_validated_r', spy)
    brainscore.score(_model(), _benchmark(packaged), check_mem=False)
    assert len(seen) == 2
    for folds, groups in seen:
        for train, test in folds:
            assert not set(groups[train]) & MADE_UP_TEST
            assert set(groups[test]) <= MADE_UP_TEST


def test_a_model_without_a_cortex_layer_is_not_applicable_before_any_data_is_read(tmp_path):
    model = BrainScoreModel('made-up-mario', region_layer_map={'V1': 'early'}, extractor={'vision': _Extractor()})
    score = brainscore.score(model, _benchmark(tmp_path / 'nothing-packaged-here'), check_mem=False)
    assert np.isnan(float(score))
    assert 'neural:cortex' in str(score.attrs)


def test_more_than_12_layers_are_refused(packaged):
    with pytest.raises(ValueError, match='at most 12'):
        brainscore.score(_model(cortex=tuple(f'layer{i}' for i in range(13))), _benchmark(packaged), check_mem=False)


def test_missing_media_name_the_command_that_makes_them(packaged):
    (packaged / 'sub-02' / 'media' / 'media.csv').unlink()
    with pytest.raises(Exception, match='data_packaging.mario'):
        brainscore.score(_model(), _benchmark(packaged), check_mem=False)


def test_the_registered_benchmark_asks_cortex_on_the_play_and_holds_out_about_a_fifth_of_the_runs():
    benchmark = brainscore.load_benchmark('Paugam2025-fMRI-mario-cortex')
    assert benchmark.requested_output_channels == {'neural:cortex'}
    assert {'vision:video', 'audio'} <= benchmark.required_input_channels
    by_participant = pd.Series([label.split('_')[0] for label in TEST_RUNS]).value_counts()
    assert set(by_participant.index) == {'sub-01', 'sub-02', 'sub-03', 'sub-05', 'sub-06'}
    assert (by_participant >= 10).all()


def _buttons(root):
    return Paugam2025MarioButtons(('sub-01', 'sub-02'), identifier='Paugam2025-behavior-mario-buttons', root=root)


def test_a_readout_on_layers_that_carry_the_play_names_the_held_buttons_above_chance(packaged, monkeypatch):
    monkeypatch.setattr(mario_benchmark, 'FIT_EVERY', 1)    # 300 fitting TRs here; the real runs give thousands
    score = brainscore.score(_model(), _buttons(packaged), check_mem=False)
    assert 'RIGHT' in score.attrs['choices'] and score.attrs['choices'][-1] == 'other'
    assert float(score) > score.attrs['chance_most_frequent'] + 0.1
    facts = score.attrs['per_participant']['sub-01']
    assert facts['n_test_trs'] == 2 * 50 and facts['n_fitting_trs'] == 6 * 50
    assert score.attrs['n_trials'] == 200 and score.attrs['answer_policy'] and score.attrs['result_validity'] == 'valid'


def test_the_chance_floor_lands_at_the_most_frequent_label(packaged):
    assert _buttons(packaged).execution_plan.metric_category == 'behavioral'
    score = brainscore.score(brainscore.load_model('chance-baseline'), _buttons(packaged), check_mem=False)
    assert float(score) == pytest.approx(score.attrs['chance_training_majority'])     # the first choice, every TR
    assert score.attrs['chance_training_majority'] <= score.attrs['chance_most_frequent']
    assert score.attrs['chance_uniform'] == 1 / len(score.attrs['choices'])


def test_the_registered_buttons_benchmark_asks_a_choice_on_the_play():
    benchmark = brainscore.load_benchmark('Paugam2025-behavior-mario-buttons')
    assert benchmark.requested_output_channels == {'behavior:choice'}
    assert 'vision:video' in benchmark.required_input_channels
