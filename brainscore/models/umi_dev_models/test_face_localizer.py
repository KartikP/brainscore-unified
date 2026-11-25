"""The face localizer on a small synthetic subject: units that respond more to faces are found on the
select half and still prefer faces on the held-out half; the matched random units do not."""
import numpy as np
import pytest

from brainscore import BrainScoreModel
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from brainscore.models.umi_dev_models.face_localizer import face_localizer

N_UNITS, FACE_UNITS = 40, (3, 11, 17, 29)
CATEGORIES = {'adult': 'faces', 'child': 'faces', 'car': 'objects', 'house': 'places'}


def _stimuli(tmp_path, per_subcategory=12):
    rows = [{'stimulus_id': f'{sub}-{n}', 'subcategory': sub, 'category': category, 'number': n,
             'image_path': str(tmp_path / f'{sub}-{n}.png')}
            for sub, category in CATEGORIES.items() for n in range(1, per_subcategory + 1)]
    for row in rows:
        open(row['image_path'], 'wb').close()
    stimuli = StimulusSet(rows)
    stimuli.stimulus_paths = {row['stimulus_id']: row['image_path'] for row in rows}
    stimuli.identifier = 'synthetic-floc'
    return stimuli


class _Extractor:
    """One layer of N_UNITS units: noise, plus 3 for the face units on a face picture."""
    identifier = 'synthetic-vision'

    def __init__(self, stimuli):
        self.category = dict(zip(stimuli['stimulus_id'], stimuli['category']))

    def __call__(self, stimuli, layers):
        ids = list(stimuli['stimulus_id'])
        rng = np.random.RandomState(len(ids))
        values = rng.normal(size=(len(ids), N_UNITS))
        faces = np.array([self.category[i] == 'faces' for i in ids])
        values[np.ix_(faces, FACE_UNITS)] += 3.0
        return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'stimulus_index': ('presentation', list(range(len(ids)))),
            'neuroid_id': ('neuroid', [f'u{j}' for j in range(N_UNITS)]),
            'neuroid_num': ('neuroid', list(range(N_UNITS))), 'layer': ('neuroid', [layers[0]] * N_UNITS)})


@pytest.fixture
def setup(tmp_path):
    stimuli = _stimuli(tmp_path)
    subject = BrainScoreModel('synthetic-face-localizer', region_layer_map={'FFA': 'blocks.9'},
                              extractor={'vision': _Extractor(stimuli)})
    return subject, stimuli


def test_the_selected_units_prefer_faces_on_the_held_out_half(setup):
    subject, stimuli = setup
    result = face_localizer(subject, stimuli, n_units=len(FACE_UNITS))
    assert result['units'].layer == 'blocks.9'
    assert sorted(result['units'].indices) == sorted(FACE_UNITS)
    held_out = result['d_test'][result['units'].indices]
    random = result['d_test'][result['control'].indices]
    assert held_out.min() > 1.0 and abs(random.mean()) < 1.0


def test_the_control_matches_the_selection_in_layer_and_size(setup):
    subject, stimuli = setup
    result = face_localizer(subject, stimuli, n_units=6, seed=1)
    units, control = result['units'], result['control']
    assert control.layer == units.layer and len(control.indices) == len(units.indices) == 6
    assert set(control.indices) <= set(range(N_UNITS))


def test_a_cross_check_set_gives_each_unit_a_d(setup, tmp_path):
    subject, stimuli = setup
    result = face_localizer(subject, stimuli, n_units=len(FACE_UNITS), cross_check=(stimuli, ('faces',), 'category'))
    assert result['d_cross'].shape == (N_UNITS,)
    assert result['d_cross'][list(FACE_UNITS)].min() > 1.0
