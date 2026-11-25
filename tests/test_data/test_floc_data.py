"""The fLoc stimulus set read from a checkout of the authors' repository (a folder of empty files here)."""
import pytest

from brainscore.data.floc.data import CATEGORIES, IMAGES_PER_SUBCATEGORY, load_stimulus_set


def _checkout(root, per_subcategory=IMAGES_PER_SUBCATEGORY):
    for subcategory in (*CATEGORIES, 'scrambled'):
        folder = root / 'stimuli' / subcategory
        folder.mkdir(parents=True)
        for number in range(1, per_subcategory + 1):
            (folder / f'{subcategory}-{number}.jpg').touch()
    return root


def test_the_ten_subcategories_are_read_with_their_condition_and_number(tmp_path):
    stimuli = load_stimulus_set(_checkout(tmp_path))
    assert len(stimuli) == len(CATEGORIES) * IMAGES_PER_SUBCATEGORY
    assert 'scrambled' not in set(stimuli['subcategory'])
    faces = stimuli[stimuli['category'] == 'faces']
    assert set(faces['subcategory']) == {'adult', 'child'}
    assert set(stimuli['number']) == set(range(1, IMAGES_PER_SUBCATEGORY + 1))
    first = stimuli.iloc[0]
    assert stimuli.stimulus_paths[first['stimulus_id']] == first['image_path']


def test_a_checkout_with_missing_images_is_refused(tmp_path):
    with pytest.raises(ValueError, match='Next step'):
        load_stimulus_set(_checkout(tmp_path, per_subcategory=3))
