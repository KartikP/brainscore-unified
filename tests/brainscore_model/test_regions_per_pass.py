"""``BRAINSCORE_REGIONS_PER_PASS`` selects the extraction plan of a session; scores never move.

``all`` (default): one model pass records every requested region of an
extractor that declares ``independent_layers``. ``one``: one pass per region.
Each channel's payload is identical under both values, for every wrapper that
declares it, in a table and in a stream. The value is read when a session
starts, so a benchmark that sets it for its own run and restores it leaves the
next run unchanged.
"""
import os

import numpy as np
import pytest

from brainscore import BrainScoreModel
from brainscore_core.streaming_helpers import stimulus_session

from tests.brainscore_model import tiny_wrappers
from tests.brainscore_model.test_one_call_per_extractor import assert_same_output

pytestmark = pytest.mark.unit
pytest.importorskip('cv2')

VARIABLE = 'BRAINSCORE_REGIONS_PER_PASS'
REGIONS = ['V1', 'V4', 'IT', 'A1']


def _counted(kind, folder):
    """A four-region model on the tiny network of ``kind``, its stimuli, the list its passes go to
    and the network."""
    wrapper, stimuli, regions, network = tiny_wrappers.FACTORIES[kind](folder)
    passes = []
    network.register_forward_pre_hook(lambda module, args: passes.append(1))
    model = BrainScoreModel(f'tiny-{kind}', extractor={tiny_wrappers.MODALITY[kind]: wrapper},
                            region_layer_map=regions)
    return model, stimuli, passes, network


def _session(model, stimuli, streaming=False):
    session = stimulus_session(stimuli, [f'neural:{region}' for region in REGIONS], streaming=streaming)
    model.participate(session)
    return session


def _events(session, region):
    return [(e.t_ms, e.meta, e.payload) for e in session.emitted if e.channel == f'neural:{region}']


@pytest.mark.parametrize('streaming', [False, True], ids=['table', 'stream'])
@pytest.mark.parametrize('kind', tiny_wrappers.KINDS)
def test_both_values_give_identical_channels(tmp_path, monkeypatch, kind, streaming):
    monkeypatch.delenv(VARIABLE, raising=False)
    default = _session(*_counted(kind, tmp_path)[:2], streaming=streaming)
    monkeypatch.setenv(VARIABLE, 'one')
    per_region = _session(*_counted(kind, tmp_path)[:2], streaming=streaming)
    for region in REGIONS:
        mine, theirs = _events(default, region), _events(per_region, region)
        assert len(mine) == len(theirs) > 0
        for (t_a, meta_a, a), (t_b, meta_b, b) in zip(mine, theirs):
            assert t_a == t_b and meta_a == meta_b
            assert_same_output(a, b)


@pytest.mark.parametrize('kind', ['video', 'audio'])
def test_four_regions_take_one_pass_per_batch_by_default_and_four_with_one(tmp_path, monkeypatch, kind):
    monkeypatch.delenv(VARIABLE, raising=False)
    model, stimuli, passes, network = _counted(kind, tmp_path)
    _session(model, stimuli)
    assert len(passes) == len(stimuli)                  # batch size 1: one pass per stimulus
    monkeypatch.setenv(VARIABLE, 'one')
    model, stimuli, passes, network = _counted(kind, tmp_path)
    _session(model, stimuli)
    assert len(passes) == 4 * len(stimuli)


@pytest.mark.parametrize('kind', ['video', 'audio'])
def test_each_region_of_a_shared_pass_equals_the_region_alone(tmp_path, monkeypatch, kind):
    monkeypatch.delenv(VARIABLE, raising=False)
    joint = _session(*_counted(kind, tmp_path)[:2])
    for region in REGIONS:
        model, stimuli, _, _ = _counted(kind, tmp_path)
        alone = stimulus_session(stimuli, [f'neural:{region}'])
        model.participate(alone)
        assert_same_output(joint.collect(f'neural:{region}'), alone.collect(f'neural:{region}'))


def _benchmark_run(kind, folder):
    """A benchmark that sets the plan for its own run and restores the earlier value."""
    previous = os.environ.get(VARIABLE)
    os.environ[VARIABLE] = 'one'
    try:
        model, stimuli, passes, network = _counted(kind, folder)
        _session(model, stimuli)
        return passes
    finally:
        if previous is None:
            del os.environ[VARIABLE]
        else:
            os.environ[VARIABLE] = previous


def test_a_value_set_and_restored_by_one_run_does_not_change_the_next(tmp_path, monkeypatch):
    monkeypatch.delenv(VARIABLE, raising=False)
    assert len(_benchmark_run('video', tmp_path)) == 12
    assert VARIABLE not in os.environ
    model, stimuli, passes, network = _counted('video', tmp_path)
    _session(model, stimuli)
    assert len(passes) == 3


def test_the_value_is_read_once_when_the_session_starts(tmp_path, monkeypatch):
    monkeypatch.delenv(VARIABLE, raising=False)
    model, stimuli, passes, network = _counted('audio', tmp_path)
    network.register_forward_hook(
        lambda *args: monkeypatch.setenv(VARIABLE, 'one'))         # changed during the session
    _session(model, stimuli, streaming=True)
    assert len(passes) == len(stimuli)                  # the plan of the start holds for every row


def test_an_unknown_value_is_refused_before_any_pass(tmp_path, monkeypatch):
    monkeypatch.setenv(VARIABLE, 'two')
    model, stimuli, passes, network = _counted('audio', tmp_path)
    with pytest.raises(ValueError, match=VARIABLE):
        _session(model, stimuli)
    assert passes == []
