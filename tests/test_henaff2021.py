"""Geometry, identity alignment, exclusion provenance and UMI participation."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest

from brainscore_core.streaming import StreamEvent
from brainscore_core.contract import Subject as SubjectContract
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore.experiments.henaff2021.data import PreparedStudy, load_spike_counts, read_reference
from brainscore.experiments.henaff2021.experiment import (
    compare_to_macaque, make_experiment, model_curvatures)
from brainscore.experiments.henaff2021.geometry import curvature

pytestmark = pytest.mark.unit


def test_curvature_has_geometric_meaning_and_is_coordinate_invariant():
    line = np.array([[0., 0.], [1., 0.], [2., 0.]])
    elbow = np.array([[0., 0.], [1., 0.], [1., 1.]])
    assert curvature(line) == pytest.approx(0)
    assert curvature(elbow) == pytest.approx(90)
    rotation = np.array([[0., -1.], [1., 0.]])
    assert curvature(3 * elbow @ rotation + 12) == pytest.approx(90)


def test_stride_two_weights_angles_not_phases():
    x = np.column_stack([np.arange(11), np.arange(11) ** 3])
    expected = (4 * curvature(x[::2]) + 3 * curvature(x[1::2])) / 7
    assert curvature(x, 2) == pytest.approx(expected)
    assert curvature(x, 2) != pytest.approx((curvature(x[::2]) + curvature(x[1::2])) / 2)


@pytest.mark.parametrize('values', [np.ones((11, 3)), [[0, 1], [1, 2]],
                                    [[0, 1], [1, np.nan], [2, 3]]])
def test_undefined_geometry_is_not_zero(values):
    with pytest.raises(ValueError):
        curvature(values)


@pytest.fixture
def study(tmp_path):
    frames = pd.DataFrame([dict(stimulus_id=f'f{i}', category='natural', scale=1, movie=1,
                               frame=i + 1, image_file=f'{i}.png') for i in range(11)])
    pixels = pd.DataFrame([dict(category='natural', scale=1, movie=1, stride=s,
                               archive_float_pixel_deg=20., presented_png_pixel_deg=21.)
                           for s in (1, 2)])
    reference = pd.DataFrame([dict(category='natural', scale=1, movie=1, stride=s,
                                  recording='population', included=True, macaque_relative_deg=-5.,
                                  archived_pixel_deg=20., archived_lnln_pixel_deg=20.,
                                  archived_lnln_relative_deg=-6.) for s in (1, 2)])
    return PreparedStudy(tmp_path, frames, pixels, reference, {'prepared_id': 'test'})


def assembly(ids):
    x = np.array([int(s[1:]) for s in ids], dtype=float)
    return NeuroidAssembly(np.column_stack([x, x * x]), dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', ids), 'presentation_id': ('presentation', np.arange(len(ids))),
        'neuroid_id': ('neuroid', ['a', 'b']), 'region': ('neuroid', ['V1', 'V1'])})


def test_reconstructs_movie_order_from_ids_not_presentation_order(study):
    first = model_curvatures(assembly(study.frames.stimulus_id.tolist()), study)
    second = model_curvatures(assembly(study.frames.stimulus_id.tolist()[::-1]), study)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize('ids', [[f'f{i}' for i in range(10)], ['f0'] * 11])
def test_missing_and_duplicate_presentations_fail(study, ids):
    with pytest.raises(ValueError, match='exactly once'):
        model_curvatures(assembly(ids), study)


def test_constant_model_invalidates_comparison(study):
    recording = assembly(study.frames.stimulus_id.tolist())
    recording.values[:] = 1
    model = model_curvatures(recording, study)
    assert (model.status == 'invalid').all()
    _, summary = compare_to_macaque(model, study.reference)
    assert all(row['pearson_r'] is None and row['mean_absolute_difference_deg'] is None
               for row in summary['comparisons'])


def test_reference_join_is_order_independent_and_rejects_missing_keys(study):
    model = model_curvatures(assembly(study.frames.stimulus_id.tolist()), study)
    _, expected = compare_to_macaque(model, study.reference)
    _, actual = compare_to_macaque(model, study.reference.iloc[::-1])
    assert actual == expected
    with pytest.raises(ValueError, match='same trajectory keys'):
        compare_to_macaque(model.iloc[:1], study.reference)


class Subject(SubjectContract):
    identifier = 'declared-participant'
    in_channels = frozenset({'vision', 'vision:still'})
    out_channels = frozenset({'neural:V1'})
    required_channels = frozenset()
    traits = frozenset()

    def __init__(self):
        self.resets, self.presentations = 0, []

    def reset(self):
        self.resets += 1

    def participate(self, session):
        while session.next_input() is not None:
            pass
        ids = session.stimulus_set.stimulus_id.tolist()
        self.presentations.extend(ids)
        assert session.task_context.task_type == 'passive'
        session.respond(StreamEvent('neural:V1', assembly(ids), t_ms=None))


def test_composes_with_existing_protocol_recorders_and_reset_lifecycle(study, tmp_path):
    subject = Subject()
    experiment = make_experiment(subject, study, tmp_path / 'run', repetitions=2)
    experiment.validate()
    assert subject.resets == 0 and subject.presentations == []
    result = experiment.run()
    assert subject.resets == 4 and len(subject.presentations) == 22
    assert set(result.neural('V1', repetition=0)['stimulus_id'].values) == set(study.frames.stimulus_id)
    assert subject.presentations[:11] != study.frames.stimulus_id.tolist()
    assert result.manifest['status'] == 'complete'


def test_incompatible_subject_fails_before_reset_and_output(study, tmp_path):
    subject = Subject()
    subject.out_channels = frozenset()
    with pytest.raises(ValueError):
        make_experiment(subject, study, tmp_path / 'bad').validate()
    assert subject.resets == 0 and not (tmp_path / 'bad').exists()


@pytest.fixture
def archive():
    value = os.environ.get('HENAFF2021_ARCHIVE')
    if not value:
        pytest.skip('Set HENAFF2021_ARCHIVE on a remote worker for source-data checks')
    return Path(value)


def test_archive_reproduces_published_cohort_and_medians(archive):
    reference = read_reference(archive)
    selected = reference[reference.included & (reference.stride == 2)]
    assert len(reference) == 720
    assert selected.groupby('category').size().to_dict() == {'natural': 127, 'artificial': 119}
    medians = selected.groupby('category').macaque_relative_deg.median()
    assert medians['natural'] == pytest.approx(-9.9, abs=.05)
    assert medians['artificial'] == pytest.approx(24.4, abs=.05)
    assert selected.recording.nunique() == 7


def test_spike_assembly_keeps_acquisition_repeats_and_shared_endpoints(archive):
    counts = load_spike_counts(archive, 'ay2_u003_image_sequences')
    assert counts.shape == (440, 39, 50)
    np.testing.assert_array_equal(counts.values[0], counts.values[110])
    np.testing.assert_array_equal(counts.values[10], counts.values[120])
    assert counts['stimulus_id'].values[0] == counts['stimulus_id'].values[110]
    assert counts.attrs['units'] == 'spike counts'


def test_movie_level_reference_matches_independent_published_workbook(archive):
    pytest.importorskip('openpyxl')
    reference = read_reference(archive)
    book = pd.read_excel(archive / 'sourceData/SourceData_final_submission.xlsx',
                         sheet_name='Figure7e', header=None)
    for category, neural_col, model_col in [('natural', 2, 1), ('artificial', 5, 4)]:
        selected = reference[reference.included & (reference.category == category) & (reference.stride == 2)]
        means = selected.groupby(['scale', 'movie'])[
            ['macaque_relative_deg', 'archived_lnln_relative_deg']].mean()
        # Plotting reorders movies. Check the paired values jointly, preserving
        # neural-to-model identity, not independently sorted marginal columns.
        actual = means.sort_values('macaque_relative_deg').to_numpy()
        published = book.iloc[2:, [neural_col, model_col]].astype(float).sort_values(neural_col).to_numpy()
        np.testing.assert_allclose(actual, published, atol=1e-10, rtol=0)


def test_reference_mapping_survives_csv_row_reordering(archive, tmp_path):
    from shutil import copyfile
    from brainscore.experiments.henaff2021.data import CSV_PATH, MODEL_PATH
    for name in [CSV_PATH, MODEL_PATH]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        copyfile(archive / name, tmp_path / name)
    csv = tmp_path / CSV_PATH
    pd.read_csv(csv).sample(frac=1, random_state=42).to_csv(csv, index=False)
    pd.testing.assert_frame_equal(read_reference(archive), read_reference(tmp_path))
