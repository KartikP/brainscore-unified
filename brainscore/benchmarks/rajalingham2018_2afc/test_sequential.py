"""Tests for the sequential (faithful-MTS) 2-AFC scaffolding.

No model weights, no objectome download: synthetic temp images for the
presentation helpers and a stub chooser for the model, so the presentation +
assembly-shape contract is verified locally. The real generation run (EC2)
reuses the same SequentialTwoAFCModel.process path.
"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from .montage import render_sample, compose_choice_array
from .sequential import SequentialTwoAFCModel, build_sequential_random_chooser


@pytest.fixture
def imgs(tmp_path):
    paths = {}
    for name, color in [('sample', (200, 80, 80)), ('left', (80, 200, 80)), ('right', (80, 80, 200))]:
        p = tmp_path / f'{name}.png'
        Image.new('RGB', (300, 300), color).save(p)
        paths[name] = str(p)
    return paths


# ── presentation helpers ─────────────────────────────────────────────

def test_render_sample_is_an_image(imgs):
    im = render_sample(imgs['sample'])
    assert isinstance(im, Image.Image)
    assert im.size[0] > 0 and im.size[1] > 0


def test_choice_array_is_wider_than_tall(imgs):
    im = compose_choice_array(imgs['left'], imgs['right'])
    # two tokens side by side -> wider than a single-token sample render
    assert im.size[0] > im.size[1]


def test_sample_and_choices_are_separate_images(imgs):
    """The whole point of the sequential variant: sample and choices are TWO
    images, never composited together."""
    s = render_sample(imgs['sample'])
    c = compose_choice_array(imgs['left'], imgs['right'])
    assert s.size != c.size            # distinct renders, not one montage


# ── SequentialTwoAFCModel.process ─────────────────────────────────────

def _stim():
    return pd.DataFrame([
        {'stimulus_id': 'trial_0', 'sample_render_path': 's0', 'choices_path': 'c0',
         'image_id': 'img0', 'sample_obj': 'wrench', 'dist_obj': 'rhino',
         'left_obj': 'wrench', 'right_obj': 'rhino'},
        {'stimulus_id': 'trial_1', 'sample_render_path': 's1', 'choices_path': 'c1',
         'image_id': 'img1', 'sample_obj': 'rhino', 'dist_obj': 'wrench',
         'left_obj': 'wrench', 'right_obj': 'rhino'},
    ])


def test_process_maps_side_to_object():
    """A stub that always answers LEFT -> chosen object is each row's left_obj."""
    model = SequentialTwoAFCModel('stub', choose_seq=lambda row: 'LEFT')
    asm = model.process(_stim())
    assert asm.dims == ('presentation',)
    assert list(asm.values) == ['wrench', 'wrench']     # left_obj of both rows
    names = set(asm.indexes['presentation'].names)
    for c in ('stimulus_id', 'sample_obj', 'dist_obj', 'truth'):
        assert c in names


def test_process_oracle_is_perfect():
    """A chooser that picks the side whose object equals the sample is 100% correct."""
    def oracle(row):
        return 'LEFT' if row['left_obj'] == row['sample_obj'] else 'RIGHT'
    asm = SequentialTwoAFCModel('oracle', oracle).process(_stim())
    correct = (asm.values == asm['truth'].values)
    assert correct.all()


def test_random_chooser_deterministic_and_binary():
    choose, stats = build_sequential_random_chooser(seed=0)
    outs = [choose({}) for _ in range(20)]
    assert set(outs) <= {'LEFT', 'RIGHT'}
    choose2, _ = build_sequential_random_chooser(seed=0)   # a fresh seeded chooser
    again = [choose2({}) for _ in range(20)]
    assert outs == again                                # seeded -> reproducible
