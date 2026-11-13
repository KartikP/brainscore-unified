"""Tests for the sequential (faithful-MTS) 2-AFC scaffolding.

No model weights, no objectome download: synthetic temp images for the
presentation helpers and a stub chooser for the model, so the presentation +
assembly-shape contract is verified locally. The real generation run (EC2)
runs the same answer part on a BrainScoreModel through participate.
"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from brainscore.benchmarks.rajalingham2018_2afc.montage import render_sample, compose_choice_array
from brainscore.benchmarks.rajalingham2018_2afc.score_2afc import run_trials
from brainscore.benchmarks.rajalingham2018_2afc.sequential import (
    SEQUENTIAL_INSTRUCTION, build_sequential_random_chooser)
from brainscore import BrainScoreModel
from brainscore_core.contract import Subject
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import drain_session_inputs, session_stimuli
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly


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


# ── the trials through participate ───────────────────────────────────

def _stim():
    return pd.DataFrame([
        {'stimulus_id': 'trial_0', 'sample_render_path': 's0', 'choices_path': 'c0',
         'image_id': 'img0', 'sample_obj': 'wrench', 'dist_obj': 'rhino',
         'left_obj': 'wrench', 'right_obj': 'rhino'},
        {'stimulus_id': 'trial_1', 'sample_render_path': 's1', 'choices_path': 'c1',
         'image_id': 'img1', 'sample_obj': 'rhino', 'dist_obj': 'wrench',
         'left_obj': 'wrench', 'right_obj': 'rhino'},
    ])


def _subject(choose):
    """A BrainScoreModel whose answer part is the chooser, as the driver builds it."""
    return BrainScoreModel('stub', supported_modalities={'vision'},
                           answer=lambda stimulus_row, instruction, label_set: choose(stimulus_row))


def _oracle(row):
    return 'LEFT' if row['left_obj'] == row['sample_obj'] else 'RIGHT'


def test_the_side_maps_to_the_object_on_that_side():
    """A stub that always answers LEFT -> chosen object is each row's left_obj."""
    choices = run_trials(_subject(lambda row: 'LEFT'), _stim(), SEQUENTIAL_INSTRUCTION)
    assert list(choices['choice']) == ['wrench', 'wrench']
    assert list(choices['sample_obj']) == ['wrench', 'rhino']


def test_the_oracle_is_perfect():
    """A chooser that picks the side whose object equals the sample is 100% correct."""
    choices = run_trials(_subject(_oracle), _stim(), SEQUENTIAL_INSTRUCTION)
    assert (choices['choice'] == choices['sample_obj']).all()


def test_the_chooser_sees_the_benchmark_row():
    rows = []
    run_trials(_subject(lambda row: rows.append(dict(row)) or 'LEFT'), _stim(), SEQUENTIAL_INSTRUCTION)
    assert rows == _stim().to_dict('records')


def test_random_chooser_deterministic_and_binary():
    choose, stats = build_sequential_random_chooser(seed=0)
    outs = [choose({}) for _ in range(20)]
    assert set(outs) <= {'LEFT', 'RIGHT'}
    choose2, _ = build_sequential_random_chooser(seed=0)   # a fresh seeded chooser
    again = [choose2({}) for _ in range(20)]
    assert outs == again                                # seeded -> reproducible


# ── any subject (Contract Rule 1.1) ───────────────────────────────────

def _one_hot(stimuli, sides):
    return BehavioralAssembly(
        np.array([[1.0, 0.0] if side == 'LEFT' else [0.0, 1.0] for side in sides]),
        coords={'stimulus_id': ('presentation', stimuli['stimulus_id'].values),
                'choice': ['LEFT', 'RIGHT']},
        dims=['presentation', 'choice'])


class _ParticipateOnlyChooser(Subject):
    """Declarations and participate only: no process(), no private state.

    It answers LEFT for every trial and returns only stimulus_id with each
    answer, so the driver must read the trial metadata from its own table.
    """

    identifier = 'participate-only'
    in_channels = frozenset({'vision:image'})
    out_channels = frozenset({'behavior'})

    def participate(self, session):
        stimuli = session_stimuli(session, drain_session_inputs(session))
        session.respond(StreamEvent('behavior', _one_hot(stimuli, ['LEFT'] * len(stimuli)), t_ms=None))


def test_participate_only_subject_runs_the_trials():
    choices = run_trials(_ParticipateOnlyChooser(), _stim(), SEQUENTIAL_INSTRUCTION)
    assert list(choices['choice']) == ['wrench', 'wrench']
    assert list(choices['image_id']) == ['img0', 'img1']
    assert list(choices['sample_obj']) == ['wrench', 'rhino']
    assert list(choices['dist_obj']) == ['rhino', 'wrench']


def test_answers_are_joined_by_stimulus_id_not_by_position():
    """A subject may answer the trials in any order; the driver joins by id."""
    class Reversed(_ParticipateOnlyChooser):
        def participate(self, session):
            stimuli = session_stimuli(session, drain_session_inputs(session)).iloc[::-1]
            session.respond(StreamEvent('behavior', _one_hot(stimuli, ['RIGHT'] * len(stimuli)), t_ms=None))

    choices = run_trials(Reversed(), _stim(), SEQUENTIAL_INSTRUCTION)
    assert list(choices['image_id']) == ['img1', 'img0']
    assert list(choices['choice']) == ['rhino', 'rhino']
    assert list(choices['sample_obj']) == ['rhino', 'wrench']


def test_witness_still_observes_the_participate_run():
    from brainscore.witness import Witness
    model = _subject(lambda row: 'LEFT')
    with Witness(model, label='stub') as witness:
        run_trials(model, _stim(), SEQUENTIAL_INSTRUCTION)
    assert len(witness.events) == 1
    assert witness.events[0].n_inputs == 2
    assert witness.events[0].mode == 'behavior'


def test_a_response_for_other_trials_is_refused():
    class Wrong(_ParticipateOnlyChooser):
        def participate(self, session):
            stimuli = session_stimuli(session, drain_session_inputs(session))
            answers = _one_hot(stimuli, ['LEFT', 'LEFT']).assign_coords(
                stimulus_id=('presentation', ['trial_0', 'trial_0']))
            session.respond(StreamEvent('behavior', answers, t_ms=None))
    with pytest.raises(ValueError, match='exactly once'):
        run_trials(Wrong(), _stim(), SEQUENTIAL_INSTRUCTION)
